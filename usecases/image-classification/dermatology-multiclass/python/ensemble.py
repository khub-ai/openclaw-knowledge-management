"""
ensemble.py — N-way derm-ham10000 ensemble orchestrator.

Implements the 4-round KF pipeline for fine-grained dermoscopic lesion
classification across all 7 HAM10000 classes simultaneously.

Round structure:
  Round 0   RuleEngine retrieves visual rules for this category set.
  Round 0.5 Retrieve (or generate) the N-way feature observation schema.
  Round 1   OBSERVER (VLM) fills in the feature schema from the test image.
  Round 2   MEDIATOR classifies into one of N classes using features + rules.
  Round 3   VERIFIER checks consistency against 1 reference image per class.

Key differences from 2-way ensemble.py:
  - task dict uses `category_set_id` + `categories` list (not pair_id/class_a/class_b).
  - Rule matching filters by `category_set_id` tag instead of `pair_id`.
  - Schema is cached under `{category_set_id}_schema`.
  - Verifier receives 1 few-shot image per class (N images total).
  - Result dict includes `category_set_id` instead of `pair_id`.
"""

from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# KF core + local imports
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# _DERM2_PY is appended (lowest priority) — provides rules.py and tools.py
# without shadowing local dataset.py / domain_config.py.
_DERM2_PY = _HERE.parents[1] / "dermatology" / "python"
if str(_DERM2_PY) not in sys.path:
    sys.path.append(str(_DERM2_PY))

from rules import RuleEngine, RuleMatch
from tools import ToolRegistry

import agents
from agents import (
    call_agent,
    reset_cost_tracker,
    get_cost_tracker,
    run_schema_generator,
    run_observer,
    run_mediator_classify,
    run_mediator_revise,
    run_verifier,
    run_rule_extractor,
)
from dataset import CATEGORY_SET_ID, CATEGORY_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_TAG   = "derm-ham10000"
MAX_REVISIONS = 1


# ---------------------------------------------------------------------------
# Round 0 — N-way rule matching
# ---------------------------------------------------------------------------

async def match_rules(rule_engine: RuleEngine, task: dict) -> list[RuleMatch]:
    """Retrieve active rules relevant to this N-way category set.

    Filters by `category_set_id` tag first, then uses the LLM for final
    relevance scoring.
    """
    active_task_rules = rule_engine.active_task_rules()
    if not active_task_rules:
        return []

    cset_id  = task.get("category_set_id", CATEGORY_SET_ID)
    cats     = task.get("categories", CATEGORY_NAMES)
    cats_str = ", ".join(cats)

    # Prefer rules tagged for this category set; fall back to all active rules.
    cset_rules   = [r for r in active_task_rules if cset_id in r.get("tags", [])]
    rules_to_match = cset_rules if cset_rules else active_task_rules

    rules_listing = rule_engine._format_rules_list(rules_to_match)

    user_msg = (
        f"You are a dermoscopy rule matcher for N-way classification.\n"
        f"Category set: {cset_id}\n"
        f"Candidate classes: {cats_str}\n\n"
        f"Determine which of the following visual rules are relevant for classifying "
        f"a dermoscopic image into one of the {len(cats)} classes above.\n"
        f"A rule is relevant if its FAVORS class is in the candidate set and its "
        f"pre-conditions could plausibly apply.\n\n"
        f"## Available Rules\n\n{rules_listing}\n\n"
        "Respond ONLY with this JSON:\n"
        "```json\n"
        '{"matches": [{"rule_id": "r_001", "confidence": "high"}, ...]}\n'
        "```\n"
        'If nothing matches, return: {"matches": []}'
    )

    text, _ = await call_agent("MEDIATOR", user_msg, max_tokens=4096)

    stripped = text.strip()
    if stripped.startswith("{") and "```" not in stripped:
        text = f"```json\n{stripped}\n```"

    return rule_engine.parse_match_response(text)


# ---------------------------------------------------------------------------
# Main N-way orchestrator
# ---------------------------------------------------------------------------

async def run_ensemble(
    task: dict,
    task_id: str = "unknown",
    rule_engine: Optional[RuleEngine] = None,
    tool_registry: Optional[ToolRegistry] = None,
    verbose: bool = True,
    dataset: str = "derm-ham10000",
    dataset_tag: str = DATASET_TAG,
    max_revisions: int = MAX_REVISIONS,
    test_mode: bool = False,
) -> dict:
    """Run the full N-way derm-ham10000 ensemble on a single image classification task.

    Args:
        task: Dict with keys:
            category_set_id  str         e.g. "dermatology_7class"
            categories       list[str]   ordered list of N class names
            dx_codes         list[str]   corresponding dx codes
            test_image_path  str         absolute path to the test JPEG
            test_label       str         ground truth class name
            few_shot         dict        {class_name: [path, ...]} — 1+ image per class
        task_id: Unique identifier (e.g. "dermatology_7class_mel_ISIC_0024306")
        rule_engine, tool_registry: created if None.
        verbose, dataset, dataset_tag, max_revisions, test_mode: as in 2-way.

    Returns:
        Result dict with: task_id, category_set_id, predicted_label, correct_label,
        correct, confidence, rounds_completed, duration_ms, cost_usd,
        feature_record, decision, verification, rule_ids_fired,
        input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
        api_calls, model, dataset, eliminated_classes.
    """
    if rule_engine is None:
        rule_engine = RuleEngine(dataset_tag=dataset_tag)
    if tool_registry is None:
        tool_registry = ToolRegistry(dataset_tag=dataset_tag)

    task["_task_id"]    = task_id
    cset_id             = task.get("category_set_id", CATEGORY_SET_ID)
    cats                = task.get("categories", CATEGORY_NAMES)
    correct_label       = task.get("test_label", "")

    reset_cost_tracker()
    t_start          = time.time()
    rounds_completed = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\n{'─'*60}")
    log(f"Task: {task_id}")
    log(f"Categories ({len(cats)}): {', '.join(cats)}")
    log(f"Ground truth: {correct_label}")
    log(f"Rules: {rule_engine.stats_summary()}")

    # ------------------------------------------------------------------
    # Round 0 — Rule matching
    # ------------------------------------------------------------------
    log("Round 0: matching rules...")
    matched_rules    = await match_rules(rule_engine, task)
    fired_ids        = [m.rule_id for m in matched_rules]
    rounds_completed += 1
    log(f"  Matched {len(matched_rules)} rule(s): {fired_ids}" if matched_rules else "  No rules matched.")

    # ------------------------------------------------------------------
    # Round 0.5 — Feature schema
    # ------------------------------------------------------------------
    schema_name = f"{cset_id}_schema"
    schema      = tool_registry.get_schema(schema_name)

    if schema:
        log(f"  Schema: loaded '{schema_name}' from registry ({len(schema.get('fields', []))} fields)")
    else:
        log(f"  Schema: generating '{schema_name}'...")
        schema, _ = await run_schema_generator(task, matched_rules)
        log(f"  Schema: generated {len(schema.get('fields', []))} fields")
        if not test_mode and schema.get("fields"):
            tool_registry.register(
                name=schema_name,
                code=json.dumps(schema),
                verified=True,
                tool_type="schema",
                source_task=cset_id,
                description=f"N-way feature observation schema for {cset_id} ({len(cats)} classes)",
            )

    # ------------------------------------------------------------------
    # Round 1 — OBSERVER
    # ------------------------------------------------------------------
    log("Round 1: OBSERVER filling feature schema from dermoscopic image...")
    t_r1             = time.time()
    feature_record, obs_ms = await run_observer(task, schema, matched_rules)
    rounds_completed += 1
    log(f"  Done in {time.time()-t_r1:.1f}s  |  {len(feature_record.get('features', {}))} features recorded")

    high_conf = {k: v for k, v in feature_record.get("features", {}).items()
                 if v.get("confidence", 0) >= 0.35}
    _hc_parts = [f"{k}={v['value']!r}" for k, v in list(high_conf.items())[:5]]
    log(f"  High-confidence features ({len(high_conf)}): {', '.join(_hc_parts)}")

    # ------------------------------------------------------------------
    # Round 2 — MEDIATOR (N-way)
    # ------------------------------------------------------------------
    log("Round 2: MEDIATOR classifying (N-way)...")
    t_r2                  = time.time()
    decision, mediator_text, med_ms = await run_mediator_classify(task, feature_record, matched_rules)
    rounds_completed     += 1
    log(f"  Done in {time.time()-t_r2:.1f}s")
    log(f"  Decision: {decision.get('label', '?')}  (confidence={decision.get('confidence', 0):.2f})")
    eliminated = decision.get("eliminated_classes", [])
    if eliminated:
        log(f"  Eliminated: {', '.join(eliminated)}")
    log(f"  Reasoning: {decision.get('reasoning', '')[:120]}")

    # ------------------------------------------------------------------
    # Round 3 — VERIFIER + revision loop
    # ------------------------------------------------------------------
    verification: dict = {}
    revision           = 0
    has_few_shot       = any(task.get("few_shot", {}).get(c) for c in cats)

    if has_few_shot:
        log("Round 3: VERIFIER checking consistency...")
        t_r3              = time.time()
        verification, _   = await run_verifier(task, decision, feature_record)
        rounds_completed += 1
        log(f"  Done in {time.time()-t_r3:.1f}s  |  consistent={verification.get('consistent', '?')}")

        while not verification.get("consistent", True) and revision < max_revisions:
            revision += 1
            log(f"  Revision {revision}/{max_revisions}: {verification.get('revision_signal', '')[:80]}")
            decision, mediator_text, _ = await run_mediator_revise(
                task, feature_record, matched_rules, decision, verification
            )
            log(f"  Revised decision: {decision.get('label', '?')}  "
                f"(confidence={decision.get('confidence', 0):.2f})")
            verification, _ = await run_verifier(task, decision, feature_record)
            log(f"  Re-check: consistent={verification.get('consistent', '?')}")
    else:
        log("Round 3: skipped (no few-shot images provided)")

    # ------------------------------------------------------------------
    # Compute result
    # ------------------------------------------------------------------
    predicted_label = decision.get("label", "uncertain")
    is_correct      = (predicted_label == correct_label) if correct_label else None

    ct          = get_cost_tracker()
    duration_ms = int((time.time() - t_start) * 1000)

    log(f"\n{'─'*60}")
    log(f"Result: {'CORRECT' if is_correct else 'WRONG'}  "
        f"predicted={predicted_label!r}  actual={correct_label!r}")
    log(f"Cost: ${ct.cost_usd():.4f}  |  API calls: {ct.api_calls}  |  "
        f"Duration: {duration_ms/1000:.1f}s")

    # ------------------------------------------------------------------
    # Post-task learning (skipped in test_mode)
    # ------------------------------------------------------------------
    if not test_mode and correct_label:
        for m in matched_rules:
            if is_correct:
                rule_engine.record_success(m.rule_id, task_id)
            else:
                rule_engine.record_failure(m.rule_id, task_id)

        rule_engine.increment_tasks_seen(fired_ids=set(fired_ids))

        extractor_text, _ = await run_rule_extractor(
            task, feature_record, decision, correct_label, is_correct,
            category_set_id=cset_id,
        )
        if extractor_text:
            new_rules = rule_engine.parse_mediator_rule_updates(extractor_text, task_id)
            if new_rules:
                log(f"  Extracted {len(new_rules)} new rule(s): {[r['id'] for r in new_rules]}")

        if is_correct and matched_rules:
            for m in matched_rules:
                rule = rule_engine.get(m.rule_id)
                if rule and rule.get("status") == "candidate" and rule.get("source_task") != task_id:
                    if rule_engine.promote_candidate(m.rule_id):
                        log(f"  Promoted candidate->active: {m.rule_id}")

        pruned = rule_engine.auto_deprecate()
        if pruned:
            log(f"  Auto-pruned {len(pruned)} rule(s): {pruned}")

    return {
        "task_id":               task_id,
        "category_set_id":       cset_id,
        "predicted_label":       predicted_label,
        "correct_label":         correct_label,
        "correct":               is_correct,
        "confidence":            decision.get("confidence", 0.0),
        "eliminated_classes":    decision.get("eliminated_classes", []),
        "rounds_completed":      rounds_completed,
        "revisions":             revision,
        "duration_ms":           duration_ms,
        "cost_usd":              round(ct.cost_usd(), 6),
        "input_tokens":          ct.input_tokens,
        "cache_creation_tokens": ct.cache_creation_tokens,
        "cache_read_tokens":     ct.cache_read_tokens,
        "output_tokens":         ct.output_tokens,
        "api_calls":             ct.api_calls,
        "model":                 agents._d2.ACTIVE_MODEL,
        "dataset":               dataset,
        "rule_ids_fired":        fired_ids,
        "feature_record":        {k: v for k, v in feature_record.items() if k != "raw_response"},
        "decision":              decision,
        "verification":          verification,
    }
