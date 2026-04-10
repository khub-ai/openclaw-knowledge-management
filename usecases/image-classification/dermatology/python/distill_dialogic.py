"""
distill_dialogic.py — Three-party dialogic knowledge distillation

Demonstrates that multi-round dialog between PUPIL, TUTOR, and KF
produces grounded rules that pass validation, where single-shot
elicitation fails (0/4 in elicit_from_failures.py).

Three parties:
  PUPIL  — the cheap VLM that failed (provides wrong prediction + reasoning)
  TUTOR  — Claude Opus (expert with dermoscopic subtype knowledge)
  KF     — orchestrator that steers the dialog:
           1. Surfaces PUPIL's failure + reasoning to TUTOR
           2. TUTOR authors a corrective rule (Round 1)
           3. KF immediately tests the rule on the trigger image (grounding check)
           4. If preconditions don't fire → KF feeds validator observations
              back to TUTOR with specific guidance on what failed
           5. TUTOR refines the rule (Round 2+)
           6. Once grounded → KF runs the standard pool gate
           7. If pool gate fails → contrastive tightening round

Evidence structure: each failure produces a dialog transcript showing
every KF steering move and the rule's evolution across rounds.

Usage:
  python distill_dialogic.py
  python distill_dialogic.py --max-rounds 4 --tutor-model claude-opus-4-6
  python distill_dialogic.py --failure-ids ISIC_0024410,ISIC_0024647
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000
from harness import CONFUSABLE_PAIRS

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
DEFAULT_PAIR       = "melanoma_vs_melanocytic_nevus"
DEFAULT_TUTOR      = "claude-opus-4-6"
DEFAULT_VALIDATOR  = "claude-sonnet-4-6"
DEFAULT_MAX_ROUNDS = 4
DEFAULT_VAL_PER_CLASS = 10
DEFAULT_FAILURE_IDS = [
    "ISIC_0024410",
    "ISIC_0024647",
    "ISIC_0024911",
    "ISIC_0025128",
]


# ── API keys ────────────────────────────────────────────────────────────────
def _load_api_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    if kf.exists():
        for line in kf.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") \
                        and not os.environ.get(k):
                    os.environ[k] = v


# ── Three-party prompts ─────────────────────────────────────────────────────

TUTOR_SYSTEM = """\
You are a senior dermoscopy expert and knowledge engineer.

Your role is to author visual rules that a validator model can confirm by
looking at dermoscopic images. The rules must use terminology that is
concrete and visually observable — not textbook abstractions.

A rule works when its preconditions can be reliably confirmed or denied
by a separate model examining the image. If your preconditions use terms
that are too abstract, the validator will fail to confirm them even when
the underlying feature IS present.

Output ONLY a JSON object:
{
  "rule": "When [preconditions], classify as [class].",
  "feature": "snake_case_feature_name",
  "favors": "<exact class name>",
  "confidence": "high" | "medium" | "low",
  "preconditions": [
    "Precondition 1 — concrete, visually checkable",
    "Precondition 2 — ...",
    ...
  ],
  "rationale": "Why this pattern distinguishes the two classes."
}
"""

ROUND1_PROMPT = """\
You are the TUTOR. A weaker PUPIL VLM classified this dermoscopic image
and got it WRONG.

Lesion pair: {class_a} vs {class_b}
Ground truth: {correct_label}
PUPIL's prediction: {wrong_prediction}  ← WRONG
PUPIL's reasoning: {pupil_reasoning}

Look at the image carefully and:
1. Identify the PUPIL's specific mistake
2. Author a corrective rule with preconditions that are VISUALLY CONCRETE —
   describe what you can see (colors, shapes, spatial arrangements, textures)
   rather than named dermoscopic concepts. For example:
   - GOOD: "gray-blue structureless areas visible within the lesion body"
   - LESS GOOD: "regression structures present"
   - GOOD: "multiple distinct color zones (brown, tan, gray-blue)"
   - LESS GOOD: "polychromatic pigmentation pattern"
3. Keep to 2-3 preconditions max — fewer conditions that are visually clear
   beat many conditions that are ambiguous
"""

REFINEMENT_PROMPT = """\
You are the TUTOR. This is round {round_num} of our dialog about this image.

Your previous rule did NOT pass the grounding check — a validator model
looked at this same image and could NOT confirm your preconditions.

YOUR PREVIOUS RULE:
{previous_rule}

YOUR PREVIOUS PRECONDITIONS:
{previous_preconditions}

VALIDATOR'S OBSERVATIONS (what it actually SAW in the image):
"{validator_observations}"

The validator said preconditions were {met_status}.

KF GUIDANCE:
{kf_guidance}

Please author a REVISED rule. Use the validator's own observations as
a vocabulary guide — if the validator described seeing "X", use "X" in
your preconditions rather than a synonym. Keep to 2-3 preconditions max.
"""

TIGHTEN_PROMPT = """\
You are the TUTOR. Your rule fires on the trigger image (good!) but
also fired on {n_fp} images from the opposite class (bad).

YOUR RULE:
{rule_text}

PRECONDITIONS:
{preconditions}

TRUE POSITIVES (correctly identified {favors}):
{tp_observations}

FALSE POSITIVES (wrongly fired on {opposing_class}):
{fp_observations}

{extra_guidance}

What single additional precondition would exclude the false positives
while keeping the true positives? The precondition MUST still be true
for the trigger image. Describe it in concrete visual terms — something
a validator could confirm or deny by looking at an image.

Reply with a JSON object:
{{
  "tightening_precondition": "the new precondition in concrete visual terms",
  "rationale": "why this separates TP from FP"
}}
"""


# ── Core dialogic loop ───────────────────────────────────────────────────────

async def run_dialogic_distillation(
    image_path: str,
    image_id: str,
    correct_label: str,
    wrong_prediction: str,
    pupil_reasoning: str,
    pair_info: dict,
    tutor_model: str,
    validator_model: str,
    max_rounds: int,
    pool_images: list,
) -> dict:
    """Run multi-round dialogic distillation for a single failure image.

    Returns a complete transcript dict with per-round evidence.
    """
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]
    opposing = class_a if correct_label != class_a else class_b

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
        console.print(f"\n  [bold]Round {round_num}[/bold]", style="cyan")
        round_record = {"round": round_num, "party_actions": []}

        # ── TUTOR turn ──────────────────────────────────────────────────
        if round_num == 1:
            prompt_text = ROUND1_PROMPT.format(
                class_a=class_a,
                class_b=class_b,
                correct_label=correct_label,
                wrong_prediction=wrong_prediction,
                pupil_reasoning=pupil_reasoning or "(not available)",
            )
        else:
            prev = transcript["rounds"][-1]
            prev_rule = active_rule
            prev_obs = prev.get("validator_observations", "")
            prev_met = "MET" if prev.get("fires_on_trigger") else "NOT MET"

            # KF steering: generate specific guidance based on what went wrong
            kf_guidance = _generate_kf_guidance(prev_rule, prev, round_num)

            prompt_text = REFINEMENT_PROMPT.format(
                round_num=round_num,
                previous_rule=prev_rule.get("rule", ""),
                previous_preconditions="\n".join(
                    f"  - {p}" for p in prev_rule.get("preconditions", [])),
                validator_observations=prev_obs,
                met_status=prev_met,
                kf_guidance=kf_guidance,
            )

        content = [
            agents._image_block(image_path),
            {"type": "text", "text": prompt_text},
        ]

        raw_text, ms = await agents.call_agent(
            "DIALOGIC_TUTOR",
            content,
            system_prompt=TUTOR_SYSTEM,
            model=tutor_model,
            max_tokens=2048,
        )

        rule = agents._parse_json_block(raw_text)
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

        console.print(f"    TUTOR → rule: [italic]{rule.get('rule', '')[:120]}[/italic]")
        for pc in rule.get("preconditions", []):
            console.print(f"      pre: {pc[:100]}")

        # ── KF grounding check ──────────────────────────────────────────
        console.print(f"    KF → grounding check on trigger image...")
        val_result, _ = await agents.run_rule_validator_on_image(
            image_path=image_path,
            ground_truth=correct_label,
            candidate_rule=rule,
            model=validator_model,
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
        console.print(f"    KF → {status}")
        console.print(f"    Validator saw: [dim]{observations[:150]}[/dim]")

        transcript["rounds"].append(round_record)

        if fires:
            transcript["grounded_at_round"] = round_num
            console.print(f"    [green]Rule grounded at round {round_num}![/green]")
            break
        elif round_num < max_rounds:
            console.print(f"    KF → steering TUTOR for round {round_num + 1}...")
        else:
            console.print(f"    [red]Max rounds ({max_rounds}) reached without grounding[/red]")

    # ── Pool gate (if grounded) ──────────────────────────────────────────
    if transcript["grounded_at_round"] is not None:
        console.print(f"\n  [bold]Pool gate[/bold] — validating on {len(pool_images)} images...")
        pool_result = await agents.validate_candidate_rule(
            candidate_rule=active_rule,
            validation_images=pool_images,
            trigger_image_path=image_path,
            trigger_correct_label=correct_label,
            model=validator_model,
        )
        transcript["pool_result"] = {
            k: v for k, v in pool_result.items()
            if k not in ("tp_cases", "fp_cases")
        }

        prec = pool_result["precision"]
        fp = pool_result["fp"]
        tp = pool_result["tp"]
        accepted = pool_result["accepted"]

        console.print(f"    TP={tp} FP={fp} precision={prec:.2f} "
                      f"{'[green]PASS[/green]' if accepted else '[red]FAIL[/red]'}")

        # ── Iterative tightening (if pool fails with FPs) ────────────────
        MAX_TIGHTEN = 3
        tighten_history = []
        if not accepted and fp > 0 and pool_result.get("tp_cases"):
            base_rule = {**active_rule}  # preserve the original grounded rule
            current_pool = pool_result

            for tighten_round in range(1, MAX_TIGHTEN + 1):
                console.print(f"\n  [bold]Tightening round {tighten_round}[/bold] "
                              f"— asking TUTOR to exclude FPs...")

                tp_obs = "\n".join(
                    f"  - {c['ground_truth']}: {c.get('observations', '')[:100]}"
                    for c in current_pool.get("tp_cases", [])[:4]
                )
                fp_obs = "\n".join(
                    f"  - {c['ground_truth']}: {c.get('observations', '')[:100]}"
                    for c in current_pool.get("fp_cases", [])[:4]
                )

                # Build extra guidance for rounds 2+
                extra = ""
                if tighten_round > 1:
                    prev_pc = tighten_history[-1].get("precondition", "")
                    prev_outcome = tighten_history[-1].get("outcome", "")
                    if prev_outcome == "over_tightened":
                        extra = (
                            f"Your PREVIOUS tightening attempt was TOO RESTRICTIVE:\n"
                            f'  "{prev_pc}"\n'
                            f"It caused the rule to stop firing on the trigger image.\n"
                            f"Try a LESS restrictive condition — something that is\n"
                            f"clearly present in melanoma images but not in benign nevi.\n"
                            f"Focus on what the FP observations describe that differs\n"
                            f"from the TP observations."
                        )
                    elif prev_outcome == "still_too_broad":
                        extra = (
                            f"Your previous tightening attempt still had too many FPs.\n"
                            f"Try a DIFFERENT distinguishing feature entirely."
                        )

                tighten_text = TIGHTEN_PROMPT.format(
                    rule_text=base_rule.get("rule", ""),
                    preconditions="\n".join(
                        f"  - {p}" for p in base_rule.get("preconditions", [])),
                    n_fp=current_pool.get("fp", fp),
                    favors=base_rule.get("favors", correct_label),
                    opposing_class=opposing,
                    tp_observations=tp_obs or "  (none)",
                    fp_observations=fp_obs or "  (none)",
                    extra_guidance=extra,
                )

                tighten_content = [
                    agents._image_block(image_path),
                    {"type": "text", "text": tighten_text},
                ]

                tighten_raw, _ = await agents.call_agent(
                    "DIALOGIC_TUTOR",
                    tighten_content,
                    system_prompt=TUTOR_SYSTEM,
                    model=tutor_model,
                    max_tokens=1024,
                )

                tighten_parsed = agents._parse_json_block(tighten_raw)
                new_pc = (tighten_parsed or {}).get("tightening_precondition", "")
                if not new_pc:
                    console.print("    TUTOR could not propose a tightening condition.")
                    tighten_history.append({"round": tighten_round,
                                            "outcome": "no_proposal"})
                    break

                console.print(f"    TUTOR → tighten: [italic]{new_pc[:120]}[/italic]")

                tightened_rule = {
                    **base_rule,
                    "preconditions": base_rule["preconditions"] + [new_pc],
                    "rule": base_rule["rule"].rstrip(".") + f"; plus {new_pc}.",
                }

                # Grounding check: does tightened rule still fire on trigger?
                console.print(f"    KF → grounding check on tightened rule...")
                trig_val, _ = await agents.run_rule_validator_on_image(
                    image_path=image_path,
                    ground_truth=correct_label,
                    candidate_rule=tightened_rule,
                    model=validator_model,
                )

                if not trig_val.get("precondition_met", False):
                    console.print(f"    [yellow]Over-tightened — rule no longer fires "
                                  f"on trigger. Trying again...[/yellow]")
                    console.print(f"    Validator: [dim]{trig_val.get('observations', '')[:120]}[/dim]")
                    tighten_history.append({
                        "round": tighten_round,
                        "precondition": new_pc,
                        "outcome": "over_tightened",
                        "validator_observations": trig_val.get("observations", ""),
                    })
                    continue

                # Pool validation on tightened rule
                pool_result2 = await agents.validate_candidate_rule(
                    candidate_rule=tightened_rule,
                    validation_images=pool_images,
                    trigger_image_path=image_path,
                    trigger_correct_label=correct_label,
                    model=validator_model,
                )

                prec2 = pool_result2["precision"]
                fp2 = pool_result2["fp"]
                tp2 = pool_result2["tp"]
                accepted2 = pool_result2["accepted"]

                console.print(f"    Tightened: TP={tp2} FP={fp2} precision={prec2:.2f} "
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
                    # Update pool for next contrastive round
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


def _generate_kf_guidance(prev_rule: dict, prev_round: dict,
                          round_num: int) -> str:
    """Generate KF's steering guidance for the next TUTOR round.

    This is where KF adds value as orchestrator — it doesn't just relay
    the validator's observations, it diagnoses *why* the rule didn't fire
    and gives the TUTOR targeted advice.
    """
    obs = prev_round.get("validator_observations", "").lower()
    preconditions = prev_rule.get("preconditions", [])

    guidance_parts = []

    # General vocabulary alignment advice
    guidance_parts.append(
        "The validator model describes images differently than you might. "
        "Use the EXACT phrases from the validator's observations where possible."
    )

    # Check if validator mentioned specific features
    if "not" in obs or "no " in obs or "absence" in obs:
        guidance_parts.append(
            "The validator explicitly noted the ABSENCE of certain features. "
            "Your preconditions may reference features that are genuinely not "
            "visible at the resolution/quality of this image."
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
            "in this image that distinguishes it from a benign mole?"
        )

    return "\n".join(f"- {g}" for g in guidance_parts)


# ── CLI + main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Three-party dialogic knowledge distillation")
    p.add_argument("--failure-ids", default="",
                   help="Comma-separated ISIC IDs (default: 4 canonical failures)")
    p.add_argument("--tutor-model", default=DEFAULT_TUTOR)
    p.add_argument("--validator-model", default=DEFAULT_VALIDATOR)
    p.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--val-per-class", type=int, default=DEFAULT_VAL_PER_CLASS)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--pair", default=DEFAULT_PAIR)
    p.add_argument("--output", default="distill_dialogic_session.json")
    return p.parse_args()


async def main():
    args = parse_args()
    _load_api_keys()

    failure_ids = (args.failure_ids.split(",") if args.failure_ids
                   else DEFAULT_FAILURE_IDS)

    pair_info = next(p for p in CONFUSABLE_PAIRS if p["pair_id"] == args.pair)
    label_mel = pair_info["class_a"]   # "Melanoma"
    label_nv  = pair_info["class_b"]   # "Melanocytic Nevus"

    console.rule("[bold]Three-Party Dialogic Distillation[/bold]")
    console.print(f"  TUTOR:     [cyan]{args.tutor_model}[/cyan]")
    console.print(f"  Validator: [cyan]{args.validator_model}[/cyan]")
    console.print(f"  Max rounds: {args.max_rounds}")
    console.print(f"  Pool size/class: {args.val_per_class}")
    console.print(f"  Failures: {failure_ids}")

    # Load dataset
    console.print(f"\n[dim]Loading HAM10000...[/dim]")
    ds = load_ham10000(args.data_dir)

    # Build image_id → path map
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0)
               + ds.sample_images("mel", 500, split="train", seed=0))
    all_nv = (ds.sample_images("nv", 500, split="test", seed=0)
              + ds.sample_images("nv", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path)
               for img in all_mel + all_nv}

    # Sample validation pool (seed=42, balanced)
    pool_mel = [(str(img.file_path), label_mel)
                for img in ds.sample_images(
                    pair_info["dx_a"], args.val_per_class,
                    split="train", seed=42)]
    pool_nv = [(str(img.file_path), label_nv)
               for img in ds.sample_images(
                   pair_info["dx_b"], args.val_per_class,
                   split="train", seed=42)]
    pool_images = pool_mel + pool_nv
    console.print(f"  Pool: {len(pool_mel)} {label_mel} + "
                  f"{len(pool_nv)} {label_nv} = {len(pool_images)}")

    # Process each failure
    all_transcripts = []
    for fid in failure_ids:
        path = img_map.get(fid)
        if not path:
            console.print(f"\n[red]Image not found: {fid}[/red]")
            continue

        console.rule(f"[bold]{fid}[/bold]")

        transcript = await run_dialogic_distillation(
            image_path=path,
            image_id=fid,
            correct_label=label_mel,
            wrong_prediction=label_nv,
            pupil_reasoning="(cheap model predicted Melanocytic Nevus)",
            pair_info=pair_info,
            tutor_model=args.tutor_model,
            validator_model=args.validator_model,
            max_rounds=args.max_rounds,
            pool_images=pool_images,
        )
        all_transcripts.append(transcript)

    # ── Summary ──────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold]Summary[/bold]")

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Image ID")
    tbl.add_column("Grounded?")
    tbl.add_column("Round")
    tbl.add_column("Pool")
    tbl.add_column("Outcome")

    n_grounded = 0
    n_accepted = 0
    for t in all_transcripts:
        grounded = t["grounded_at_round"] is not None
        n_grounded += grounded
        pool = t.get("pool_result") or {}
        pool_t = t.get("pool_result_after_tighten")
        # Use tightened result if it exists and passed
        if pool_t and pool_t.get("accepted"):
            pool = pool_t
        accepted = pool.get("accepted", False)
        n_accepted += accepted

        tbl.add_row(
            t["image_id"],
            "[green]Yes[/green]" if grounded else "[red]No[/red]",
            str(t["grounded_at_round"] or "—"),
            (f"TP={pool.get('tp',0)} FP={pool.get('fp',0)} "
             f"prec={pool.get('precision',0):.2f}")
            if pool else "—",
            ("[green]ACCEPTED[/green]" if accepted
             else "[yellow]grounded[/yellow]" if grounded
             else "[red]not grounded[/red]"),
        )

    console.print(tbl)

    # Comparison with single-shot
    console.print(
        f"\n  Single-shot baseline (elicit_from_failures.py): "
        f"[red]0/{len(all_transcripts)} grounded, 0 accepted[/red]"
    )
    console.print(
        f"  Dialogic (this run): "
        f"[cyan]{n_grounded}/{len(all_transcripts)} grounded, "
        f"{n_accepted} accepted[/cyan]"
    )

    # Save session
    session = {
        "tutor_model": args.tutor_model,
        "validator_model": args.validator_model,
        "max_rounds": args.max_rounds,
        "pool_size_per_class": args.val_per_class,
        "pair": args.pair,
        "failure_ids": failure_ids,
        "transcripts": all_transcripts,
        "summary": {
            "total_failures": len(all_transcripts),
            "grounded": n_grounded,
            "accepted": n_accepted,
            "single_shot_grounded": 0,
            "single_shot_accepted": 0,
        },
    }

    out_path = _HERE / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    console.print(f"\n  Session saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
