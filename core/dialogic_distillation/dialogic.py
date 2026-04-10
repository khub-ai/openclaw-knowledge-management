"""
dialogic.py — Multi-round dialogic distillation protocol.

Three parties collaborate:
  PUPIL  — the cheap VLM that failed (provides wrong prediction + reasoning)
  TUTOR  — the expert model (authors corrective rules)
  KF     — the orchestrator (steers dialog, validates, registers rules)

The protocol:
  1. PUPIL fails on a case
  2. KF shows the failure to TUTOR
  3. TUTOR proposes a corrective rule (Round 1)
  4. KF immediately tests the rule on the trigger image (grounding check)
  5. If preconditions don't fire -> KF feeds validator observations
     back to TUTOR with specific guidance
  6. TUTOR refines the rule (Round 2+)
  7. Once grounded -> KF runs the pool gate
  8. If pool fails with FPs -> contrastive tightening rounds
"""
from __future__ import annotations
from typing import Callable, Optional

from .protocols import DomainConfig
from .constants import DEFAULT_MAX_TIGHTENING_ROUNDS
from . import agents as _agents
from . import prompts as _prompts


async def run_dialogic_distillation(
    image_path: str,
    image_id: str,
    correct_label: str,
    wrong_prediction: str,
    pupil_reasoning: str,
    pair_info: dict,
    config: DomainConfig,
    tutor_model: str,
    validator_model: str,
    max_rounds: int,
    pool_images: list,
    max_tightening_rounds: int = DEFAULT_MAX_TIGHTENING_ROUNDS,
    call_agent_fn: Callable | None = None,
    console=None,
) -> dict:
    """Run multi-round dialogic distillation for a single failure image.

    Returns a complete transcript dict with per-round evidence.
    """
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]
    opposing = class_a if correct_label != class_a else class_b

    # Optional rich console for progress output
    _print = console.print if console else lambda *a, **kw: None

    transcript = {
        "image_id": image_id,
        "correct_label": correct_label,
        "wrong_prediction": wrong_prediction,
        "rounds": [],
        "final_rule": None,
        "grounded_at_round": None,
        "pool_result": None,
        "pool_result_after_tighten": None,
        "outcome": "pending",
    }

    active_rule = None

    for round_num in range(1, max_rounds + 1):
        _print(f"\n  [bold]Round {round_num}[/bold]", style="cyan")
        round_record = {"round": round_num, "party_actions": []}

        # -- TUTOR turn --
        if round_num == 1:
            # Build vocabulary examples block
            vocab_lines = ""
            if config.good_vocabulary_examples or config.bad_vocabulary_examples:
                parts = []
                for ex in config.good_vocabulary_examples:
                    parts.append(f"   - GOOD: \"{ex}\"")
                for ex in config.bad_vocabulary_examples:
                    parts.append(f"   - BAD: \"{ex}\"")
                vocab_lines = "\n".join(parts) + "\n"

            prompt_text = _prompts.ROUND1_PROMPT.format(
                class_a=class_a,
                class_b=class_b,
                correct_label=correct_label,
                wrong_prediction=wrong_prediction,
                pupil_reasoning=pupil_reasoning or "(not available)",
                item_noun=config.item_noun,
                class_noun=config.class_noun,
                vocab_examples=vocab_lines,
            )
        else:
            prev = transcript["rounds"][-1]
            prev_rule = active_rule
            prev_obs = prev.get("validator_observations", "")
            prev_met = "MET" if prev.get("fires_on_trigger") else "NOT MET"

            kf_guidance = generate_kf_guidance(
                prev_rule, prev, round_num, config
            )

            prompt_text = _prompts.REFINEMENT_PROMPT.format(
                round_num=round_num,
                previous_rule=prev_rule.get("rule", ""),
                previous_preconditions="\n".join(
                    f"  - {p}" for p in prev_rule.get("preconditions", [])),
                validator_observations=prev_obs,
                met_status=prev_met,
                kf_guidance=kf_guidance,
                item_noun=config.item_noun,
            )

        content = [
            _agents.image_block(image_path),
            {"type": "text", "text": prompt_text},
        ]

        raw_text, ms = await (_agents._get_default_call_agent()
                              if call_agent_fn is None else call_agent_fn)(
            "DIALOGIC_TUTOR",
            content,
            system_prompt=_prompts.dialogic_tutor_system(config),
            model=tutor_model,
            max_tokens=2048,
        )

        rule = _agents.parse_json_block(raw_text)
        if not rule or "preconditions" not in rule:
            rule = {"rule": raw_text, "feature": "unknown",
                    "favors": correct_label, "confidence": "low",
                    "preconditions": [], "rationale": ""}
        rule["raw_response"] = raw_text
        active_rule = rule

        round_record["party_actions"].append({
            "party": "TUTOR",
            "action": "author_rule" if round_num == 1 else "refine_rule",
            "rule": rule.get("rule", ""),
            "preconditions": rule.get("preconditions", []),
            "rationale": rule.get("rationale", ""),
        })

        _print(f"    TUTOR -> rule: [italic]{rule.get('rule', '')[:120]}[/italic]")
        for pc in rule.get("preconditions", []):
            _print(f"      pre: {pc[:100]}")

        # -- KF grounding check --
        _print("    KF -> grounding check on trigger image...")
        val_result, _ = await _agents.run_rule_validator_on_image(
            image_path=image_path,
            ground_truth=correct_label,
            candidate_rule=rule,
            config=config,
            model=validator_model,
            call_agent_fn=call_agent_fn,
        )

        fires = val_result.get("precondition_met", False)
        observations = val_result.get("observations", "")

        round_record["party_actions"].append({
            "party": "KF",
            "action": "grounding_check",
            "fires_on_trigger": fires,
            "validator_observations": observations,
        })
        round_record["fires_on_trigger"] = fires
        round_record["validator_observations"] = observations

        status = "[green]FIRES[/green]" if fires else "[red]DOES NOT FIRE[/red]"
        _print(f"    KF -> {status}")
        _print(f"    Validator saw: [dim]{observations[:150]}[/dim]")

        transcript["rounds"].append(round_record)

        if fires:
            transcript["grounded_at_round"] = round_num
            _print(f"    [green]Rule grounded at round {round_num}![/green]")
            break
        elif round_num < max_rounds:
            _print(f"    KF -> steering TUTOR for round {round_num + 1}...")
        else:
            _print(f"    [red]Max rounds ({max_rounds}) reached without grounding[/red]")

    # -- Pool gate (if grounded) --
    if transcript["grounded_at_round"] is not None:
        _print(f"\n  [bold]Pool gate[/bold] -- validating on {len(pool_images)} images...")
        pool_result = await _agents.validate_candidate_rule(
            candidate_rule=active_rule,
            validation_images=pool_images,
            trigger_image_path=image_path,
            trigger_correct_label=correct_label,
            config=config,
            model=validator_model,
            call_agent_fn=call_agent_fn,
        )
        transcript["pool_result"] = {
            k: v for k, v in pool_result.items()
            if k not in ("tp_cases", "fp_cases")
        }

        prec = pool_result["precision"]
        fp = pool_result["fp"]
        tp = pool_result["tp"]
        accepted = pool_result["accepted"]

        _print(f"    TP={tp} FP={fp} precision={prec:.2f} "
               f"{'[green]PASS[/green]' if accepted else '[red]FAIL[/red]'}")

        # -- Iterative tightening (if pool fails with FPs) --
        tighten_history = []
        if not accepted and fp > 0 and pool_result.get("tp_cases"):
            current_pool = pool_result

            for tighten_round in range(1, max_tightening_rounds + 1):
                _print(f"\n  [bold]Tightening round {tighten_round}[/bold] "
                       f"-- asking TUTOR to exclude FPs...")

                tp_obs = "\n".join(
                    f"  - {c['ground_truth']}: {c.get('observations', '')[:100]}"
                    for c in current_pool.get("tp_cases", [])[:4]
                )
                fp_obs = "\n".join(
                    f"  - {c['ground_truth']}: {c.get('observations', '')[:100]}"
                    for c in current_pool.get("fp_cases", [])[:4]
                )

                extra = ""
                if tighten_round > 1:
                    prev_pc = tighten_history[-1].get("precondition", "")
                    prev_outcome = tighten_history[-1].get("outcome", "")
                    if prev_outcome == "over_tightened":
                        extra = (
                            f"Your PREVIOUS tightening attempt was TOO RESTRICTIVE:\n"
                            f'  "{prev_pc}"\n'
                            f"It caused the rule to stop firing on the trigger {config.item_noun}.\n"
                            f"Try a LESS restrictive condition — something that is\n"
                            f"clearly present in {active_rule.get('favors', '')} {config.item_noun_plural} "
                            f"but not in the opposing {config.class_noun}.\n"
                            f"Focus on what the FP observations describe that differs\n"
                            f"from the TP observations."
                        )
                    elif prev_outcome == "still_too_broad":
                        extra = (
                            f"Your previous tightening attempt still had too many FPs.\n"
                            f"Try a DIFFERENT distinguishing feature entirely."
                        )

                tighten_text = _prompts.TIGHTEN_PROMPT.format(
                    rule_text=active_rule.get("rule", ""),
                    preconditions="\n".join(
                        f"  - {p}" for p in active_rule.get("preconditions", [])),
                    n_fp=current_pool.get("fp", fp),
                    favors=active_rule.get("favors", correct_label),
                    opposing_class=opposing,
                    tp_observations=tp_obs or "  (none)",
                    fp_observations=fp_obs or "  (none)",
                    extra_guidance=extra,
                    item_noun=config.item_noun,
                    item_noun_plural=config.item_noun_plural,
                    class_noun=config.class_noun,
                )

                tighten_content = [
                    _agents.image_block(image_path),
                    {"type": "text", "text": tighten_text},
                ]

                tighten_raw, _ = await (_agents._get_default_call_agent()
                                        if call_agent_fn is None else call_agent_fn)(
                    "DIALOGIC_TUTOR",
                    tighten_content,
                    system_prompt=_prompts.dialogic_tutor_system(config),
                    model=tutor_model,
                    max_tokens=1024,
                )

                tighten_parsed = _agents.parse_json_block(tighten_raw)
                new_pc = (tighten_parsed or {}).get("tightening_precondition", "")
                if not new_pc:
                    _print("    TUTOR could not propose a tightening condition.")
                    tighten_history.append({
                        "round": tighten_round, "outcome": "no_proposal"})
                    break

                _print(f"    TUTOR -> tighten: [italic]{new_pc[:120]}[/italic]")

                tightened_rule = {
                    **active_rule,
                    "preconditions": active_rule["preconditions"] + [new_pc],
                    "rule": active_rule["rule"].rstrip(".") + f"; plus {new_pc}.",
                }

                # Grounding check on tightened rule
                _print("    KF -> grounding check on tightened rule...")
                trig_val, _ = await _agents.run_rule_validator_on_image(
                    image_path=image_path,
                    ground_truth=correct_label,
                    candidate_rule=tightened_rule,
                    config=config,
                    model=validator_model,
                    call_agent_fn=call_agent_fn,
                )

                if not trig_val.get("precondition_met", False):
                    _print(f"    [yellow]Over-tightened -- rule no longer fires "
                           f"on trigger. Trying again...[/yellow]")
                    tighten_history.append({
                        "round": tighten_round,
                        "precondition": new_pc,
                        "outcome": "over_tightened",
                        "validator_observations": trig_val.get("observations", ""),
                    })
                    continue

                # Pool validation on tightened rule
                pool_result2 = await _agents.validate_candidate_rule(
                    candidate_rule=tightened_rule,
                    validation_images=pool_images,
                    trigger_image_path=image_path,
                    trigger_correct_label=correct_label,
                    config=config,
                    model=validator_model,
                    call_agent_fn=call_agent_fn,
                )

                prec2 = pool_result2["precision"]
                fp2 = pool_result2["fp"]
                tp2 = pool_result2["tp"]
                accepted2 = pool_result2["accepted"]

                _print(f"    Tightened: TP={tp2} FP={fp2} precision={prec2:.2f} "
                       f"{'[green]PASS[/green]' if accepted2 else '[red]FAIL[/red]'}")

                tighten_history.append({
                    "round": tighten_round,
                    "precondition": new_pc,
                    "outcome": "accepted" if accepted2 else "still_too_broad",
                    "pool": {k: v for k, v in pool_result2.items()
                             if k not in ("tp_cases", "fp_cases")},
                })

                if accepted2:
                    active_rule = tightened_rule
                    accepted = True
                    break
                else:
                    current_pool = pool_result2

            transcript["tighten_history"] = tighten_history
            if accepted:
                transcript["pool_result_after_tighten"] = tighten_history[-1].get("pool")
            else:
                transcript["pool_result_after_tighten"] = (
                    tighten_history[-1].get("pool") if tighten_history else None
                )

        transcript["final_rule"] = {
            k: v for k, v in active_rule.items() if k != "raw_response"
        }
        transcript["outcome"] = "accepted" if accepted else "grounded_but_pool_failed"
    else:
        transcript["final_rule"] = {
            k: v for k, v in active_rule.items() if k != "raw_response"
        } if active_rule else None
        transcript["outcome"] = "not_grounded"

    return transcript


def generate_kf_guidance(
    prev_rule: dict,
    prev_round: dict,
    round_num: int,
    config: DomainConfig,
) -> str:
    """Generate KF's steering guidance for the next TUTOR round.

    This is where KF adds value as orchestrator — it doesn't just relay
    the validator's observations, it diagnoses *why* the rule didn't fire
    and gives the TUTOR targeted advice.
    """
    obs = prev_round.get("validator_observations", "").lower()
    preconditions = prev_rule.get("preconditions", [])

    guidance_parts = []

    guidance_parts.append(
        "The validator model describes images differently than you might. "
        "Use the EXACT phrases from the validator's observations where possible."
    )

    if "not" in obs or "no " in obs or "absence" in obs:
        guidance_parts.append(
            "The validator explicitly noted the ABSENCE of certain features. "
            "Your preconditions may reference features that are genuinely not "
            f"visible at the resolution/quality of this {config.item_noun}."
        )

    if len(preconditions) > 3:
        guidance_parts.append(
            f"You had {len(preconditions)} preconditions — too many increases "
            "the chance that one fails. Consolidate to 2-3 strong ones."
        )

    if round_num >= 3:
        guidance_parts.append(
            "We are on round 3+. Try a fundamentally different visual signal "
            "rather than rephrasing the same features. What ELSE do you see "
            f"in this {config.item_noun} that distinguishes it?"
        )

    return "\n".join(f"- {g}" for g in guidance_parts)
