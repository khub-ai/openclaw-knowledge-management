"""
patch.py — N-way dialogic patching loop for the KF dermatology-multiclass pipeline.

Workflow:
  1. Run the cheap VLM zero-shot on all N classes to identify failures.
  2. For each failure, call an expert VLM to author a corrective rule.
     The rule carries both `favors` (correct class) and `contra` (ruled-out classes).
  3. Validate the candidate rule against the labeled training pool.
  4. Register accepted rules in the N-way knowledge base.
  5. Re-run the cheap VLM with the patch rules and report improvement.

Key N-way differences from the 2-way patch.py:
  - Failures are identified in a 7-class space (not a binary pair).
  - Rule authoring receives the full category list and sets `contra` explicitly.
  - The validation pool draws images from all N classes.
  - Rule tags use `category_set_id` instead of `pair_id`.

Usage:
  # Full loop: zero-shot → patch → re-test
  python patch.py --cheap-model google/gemma-4-26b-a4b-it --expert-model claude-sonnet-4-6

  # Start from an existing zero-shot results file
  python patch.py --failures-from results_nway_baseline_gemma4_zeroshot.json \\
                  --cheap-model google/gemma-4-26b-a4b-it --expert-model claude-sonnet-4-6

  # Dry-run (author and validate only, no registration)
  python patch.py --cheap-model google/gemma-4-26b-a4b-it --dry-run
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_KF_ROOT  = _HERE.parents[4]
_DERM2_PY = _HERE.parents[1] / "dermatology" / "python"
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_DERM2_PY) not in sys.path:
    sys.path.append(str(_DERM2_PY))

import agents
from dataset import (
    load as load_ham10000,
    ALL_CLASSES,
    CATEGORY_SET_ID,
    CATEGORY_NAMES,
    DX_TO_NAME,
    NAME_TO_DX,
    DEFAULT_DATA_DIR,
)
from harness import build_tasks, task_id_for, DEFAULT_N_FEW_SHOT
from rules import RuleEngine

from rich.console import Console
from rich.table import Table

console = Console()

DATASET_TAG = "derm-ham10000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_keys() -> None:
    key_file = Path("P:/_access/Security/api_keys.env")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY") \
                   and not os.environ.get(k):
                    os.environ[k] = v


def _init_patch_rules(patch_rules_path: Path) -> RuleEngine:
    """Create a fresh, isolated N-way patch rules file."""
    if patch_rules_path.exists():
        patch_rules_path.unlink()
    return RuleEngine(str(patch_rules_path), dataset_tag=DATASET_TAG)


def _register_rule(
    rule_engine: RuleEngine,
    candidate_rule: dict,
    category_set_id: str,
    validation: dict,
    triggered_by: str,
    cheap_model: str,
    expert_model: str,
) -> bool:
    """Register an accepted N-way patch rule.

    The `contra` list is encoded in both the condition and action strings so it
    survives the RuleEngine's JSON storage and is recoverable by the Mediator formatter.
    """
    rule_text     = candidate_rule["rule"]
    favors        = candidate_rule["favors"]
    contra        = candidate_rule.get("contra", [])
    confidence    = candidate_rule.get("confidence", "medium")
    preconditions = candidate_rule.get("preconditions", [])

    # Encode preconditions in the condition string (hard-gate format)
    precond_str = "; ".join(preconditions) if preconditions else rule_text
    condition   = f"[Patch rule — {category_set_id}] {precond_str}"

    # Encode favors + contra in the action string for Mediator display
    contra_str  = f" [CONTRA: {', '.join(contra)}]" if contra else ""
    action      = (
        f"Classify as {favors}. Confidence: {confidence}. "
        f"Rule: {rule_text}{contra_str}"
    )

    result = rule_engine.add_rule(
        condition=condition,
        action=action,
        source=f"expert_vlm:{expert_model}",
        source_task=triggered_by,
        tags=[DATASET_TAG, category_set_id, f"patch:{cheap_model}",
              f"triggered_by:{triggered_by}", f"favors:{favors}"],
        lineage={"type": "new", "parent_ids": [],
                 "reason": f"nway patch rule for {category_set_id} triggered by {triggered_by}"},
        observability_filter=False,
    )

    # Attach favors / contra as live metadata on the rule dict so the Mediator
    # formatter can surface them without re-parsing the action string.
    if result is not None:
        rule = rule_engine.get(result["id"])
        if rule:
            rule["favors"] = favors
            rule["contra"] = contra
            rule_engine.save()

    return result is not None


def _run_zero_shot_baseline(
    cheap_model: str,
    data_dir: str,
    output: str,
    max_per_class: int,
) -> list[dict]:
    """Run cheap model zero-shot N-way baseline via harness.py subprocess."""
    result = subprocess.run(
        [
            sys.executable, "harness.py",
            "--all",
            "--max-per-class", str(max_per_class),
            "--mode", "test",
            "--model", cheap_model,
            "--baseline", "zero_shot",
            "--output", output,
            "--data-dir", data_dir,
        ],
        cwd=_HERE,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        console.print("[red]Zero-shot baseline run failed[/red]")
        sys.exit(1)

    out_path = _HERE / output
    if not out_path.exists():
        return []
    with open(out_path) as f:
        data = json.load(f)
    return data.get("tasks", [])


def _run_pipeline_with_patch_rules(
    cheap_model: str,
    data_dir: str,
    output: str,
    max_per_class: int,
    patch_rules_path: Path,
    failure_task_ids: list[str] | None = None,
) -> list[dict]:
    """Re-run cheap model + patch rules through the N-way KF pipeline."""
    cmd = [
        sys.executable, "harness.py",
        "--all",
        "--max-per-class", str(max_per_class),
        "--mode", "test",
        "--model", cheap_model,
        "--rules", str(patch_rules_path),
        "--output", output,
        "--data-dir", data_dir,
    ]

    task_list_path: Path | None = None
    if failure_task_ids:
        task_list_path = _HERE / "_nway_patch_rerun_tasks.json"
        with open(task_list_path, "w") as f:
            json.dump(failure_task_ids, f)
        cmd += ["--task-list", str(task_list_path)]

    result = subprocess.run(cmd, cwd=_HERE, capture_output=False, text=True)
    if result.returncode != 0:
        console.print("[red]Pipeline re-run failed[/red]")
        sys.exit(1)

    if task_list_path and task_list_path.exists():
        task_list_path.unlink()

    out_path = _HERE / output
    if not out_path.exists():
        return []
    with open(out_path) as f:
        data = json.load(f)
    return data.get("tasks", [])


def _task_from_results_entry(entry: dict, ds) -> dict | None:
    """Reconstruct an N-way task dict from a results entry + dataset."""
    task_id = entry["task_id"]
    dx      = entry.get("_dx", "")

    # task_id format: {CATEGORY_SET_ID}_{dx}_{image_id}
    # e.g. "dermatology_7class_nv_ISIC_0024392"
    # Strip the known prefix to isolate "{image_id}" cleanly.
    prefix   = f"{CATEGORY_SET_ID}_{dx}_"
    image_id = task_id[len(prefix):] if task_id.startswith(prefix) else "_".join(task_id.rsplit("_", 2)[-2:])

    # Build shared few_shot for all classes (same as harness)
    few_shot: dict[str, list[str]] = {}
    for c in ALL_CLASSES:
        imgs = ds.sample_images(c["dx"], n=DEFAULT_N_FEW_SHOT, split="train", seed=42)
        few_shot[c["name"]] = [str(img.file_path) for img in imgs]

    # Locate the image in train or test split
    for c in ALL_CLASSES:
        for split in ("test", "train"):
            for img in ds.images_for_class(c["dx"], split=split):
                if img.image_id == image_id:
                    return {
                        "category_set_id": CATEGORY_SET_ID,
                        "categories":      CATEGORY_NAMES,
                        "dx_codes":        [c["dx"] for c in ALL_CLASSES],
                        "test_image_path": str(img.file_path),
                        "test_label":      entry["correct_label"],
                        "few_shot":        few_shot,
                        "_image_id":       img.image_id,
                        "_dx":             c["dx"],
                    }
    return None


# ---------------------------------------------------------------------------
# Core N-way patch loop
# ---------------------------------------------------------------------------

async def run_patch_loop(
    failures: list[dict],
    ds,
    cheap_model: str,
    expert_model: str,
    rule_engine: RuleEngine,
    max_val_per_class: int,
    max_authoring_per_class: int,
    max_confirm_per_class: int,
    min_precision: float,
    dry_run: bool,
    validator_model: str = "",
) -> list[dict]:
    """Author, validate, and register N-way rules for each failure.

    For each failure the expert model authors a rule that:
      - `favors`: the correct class
      - `contra`: at minimum the wrong predicted class, possibly others

    Rules are written ONLY to the isolated rule_engine (patch_rules_nway.json).
    """
    patch_records: list[dict] = []
    _val_model = validator_model or expert_model

    for failure in failures:
        task_id       = failure["task_id"]
        wrong_pred    = failure["predicted_label"]
        correct_label = failure["correct_label"]

        console.rule(f"Failure: {task_id}")
        console.print(f"  Wrong: [red]{wrong_pred}[/red]  →  Correct: [green]{correct_label}[/green]")

        # Reconstruct full task dict
        task = _task_from_results_entry(failure, ds)
        if task is None:
            console.print(f"  [yellow]Could not find image for {task_id} — skipping.[/yellow]")
            patch_records.append({"task_id": task_id, "skipped": True})
            continue

        agents._set_active_model(expert_model)

        # Step A: Author rule
        console.print(f"  Calling expert VLM ({expert_model}) to author rule...")
        candidate_rule, _ = await agents.run_expert_rule_author(
            task=task,
            wrong_prediction=wrong_pred,
            correct_label=correct_label,
            model=expert_model,
        )

        if not candidate_rule or not candidate_rule.get("rule"):
            console.print("  [yellow]Expert produced no rule — skipping.[/yellow]")
            patch_records.append({"task_id": task_id, "rule_authored": False})
            continue

        console.print(f"  Rule: {candidate_rule['rule'][:120]}")
        console.print(f"  Favors: [green]{candidate_rule.get('favors', '?')}[/green]  |  "
                      f"Contra: {candidate_rule.get('contra', [])}")
        console.print(f"  Confidence: {candidate_rule.get('confidence', '?')}")

        # Step B: Rule completion (fill in implicit background conditions)
        # Build a pseudo pair_info for the completer (uses favors/wrong pair as anchor)
        pair_info_pseudo = {
            "pair_id":    CATEGORY_SET_ID,
            "class_a":    correct_label,
            "class_b":    wrong_pred,
            "category_set_id": CATEGORY_SET_ID,
            "categories": CATEGORY_NAMES,
        }
        pre_completion_rule = dict(candidate_rule)
        completed_rule, _ = await agents.run_rule_completer(
            candidate_rule=candidate_rule, pair_info=pair_info_pseudo, model=expert_model,
        )
        candidate_rule = completed_rule if completed_rule.get("rule") else candidate_rule
        added_conds = candidate_rule.get("added_conditions", [])
        if added_conds:
            console.print(f"  {len(added_conds)} background condition(s) added.")

        # Step C: Semantic validation
        console.print(f"  Running semantic validation ({_val_model})...")
        sem_val, _ = await agents.run_semantic_rule_validator(
            candidate_rule=candidate_rule, pair_info=pair_info_pseudo, model=_val_model,
        )
        sem_ok = sem_val.get("verdict", "ACCEPT") != "REJECT"
        if not sem_ok:
            console.print(f"  [yellow]Semantic: REJECT — {sem_val.get('reason', '')[:80]}[/yellow]")
            patch_records.append({
                "task_id": task_id, "rule_authored": True, "rule_accepted": False,
                "semantic_rejection": sem_val.get("reason", ""),
            })
            continue

        # Step D: Held-out gate (one image per class from train split)
        held_out_images: list[tuple[str, str]] = []
        for c in ALL_CLASSES:
            imgs = ds.sample_images(c["dx"], max_val_per_class, split="train", seed=42)
            held_out_images.extend([(str(img.file_path), c["name"]) for img in imgs])

        console.print(f"  Held-out pool: {len(held_out_images)} images ({len(ALL_CLASSES)} classes × ≤{max_val_per_class} each)")
        console.print("  Running held-out gate...")

        agents._set_active_model(_val_model)
        validation = await agents.validate_candidate_rule(
            candidate_rule=candidate_rule,
            validation_images=held_out_images,
            trigger_image_path=task["test_image_path"],
            trigger_correct_label=correct_label,
            model=_val_model,
        )

        precision         = validation["precision"]
        fires_on_trigger  = validation["fires_on_trigger"]
        accepted          = validation["accepted"] and precision >= min_precision

        console.print(
            f"  Held-out gate: TP={validation['tp']} FP={validation['fp']} "
            f"TN={validation['tn']} FN={validation['fn']} | "
            f"precision={precision:.2f} recall={validation['recall']:.2f} | "
            f"fires_on_trigger={fires_on_trigger}"
        )

        if not accepted:
            console.print("  [red]Held-out gate FAILED — rule not registered.[/red]")
            patch_records.append({
                "task_id": task_id, "rule_authored": True, "rule_accepted": False,
                "validation": validation,
            })
            continue

        # Step E: Confirmation pool (different seed, larger)
        confirm_imgs: list[tuple[str, str]] = []
        for c in ALL_CLASSES:
            imgs = ds.sample_images(c["dx"], max_confirm_per_class, split="train", seed=123)
            # Exclude anything already in held-out
            seen = {p for p, _ in held_out_images}
            confirm_imgs.extend(
                [(str(img.file_path), c["name"]) for img in imgs
                 if str(img.file_path) not in seen]
            )

        console.print(f"  Confirmation pool: {len(confirm_imgs)} fresh images — validating...")
        confirm_val = await agents.validate_candidate_rule(
            candidate_rule=candidate_rule,
            validation_images=confirm_imgs,
            trigger_image_path=task["test_image_path"],
            trigger_correct_label=correct_label,
            model=_val_model,
        )
        confirmed = confirm_val["accepted"] and confirm_val["precision"] >= min_precision
        console.print(
            f"  Confirmation: TP={confirm_val['tp']} FP={confirm_val['fp']} "
            f"TN={confirm_val['tn']} | precision={confirm_val['precision']:.2f} "
            f"→ {'[green]CONFIRMED[/green]' if confirmed else '[red]REJECTED[/red]'}"
        )

        if not confirmed:
            console.print("  Rule did not generalize — not registered.")
            patch_records.append({
                "task_id": task_id, "rule_authored": True, "rule_accepted": False,
                "held_out_validation": validation, "confirmation_validation": confirm_val,
            })
            continue

        # Step F: Register
        registered = False
        if not dry_run:
            registered = _register_rule(
                rule_engine=rule_engine,
                candidate_rule=candidate_rule,
                category_set_id=CATEGORY_SET_ID,
                validation=validation,
                triggered_by=task_id,
                cheap_model=cheap_model,
                expert_model=expert_model,
            )
            if registered:
                console.print("  [green]Rule registered in N-way patch rules file.[/green]")
            else:
                console.print("  [yellow]Rule engine rejected rule (duplicate?).[/yellow]")
        else:
            console.print("  [dim]Dry-run: rule NOT registered.[/dim]")

        patch_records.append({
            "task_id":                task_id,
            "rule_authored":          True,
            "rule_accepted":          True,
            "rule_registered":        registered,
            "candidate_rule":         {k: v for k, v in candidate_rule.items() if k != "raw_response"},
            "held_out_validation":    {k: v for k, v in validation.items()
                                       if k not in ("tp_cases", "fp_cases")},
            "confirmation_validation": {k: v for k, v in confirm_val.items()
                                        if k not in ("tp_cases", "fp_cases")},
        })

        # Step G: Immediate test — re-run cheap model on THIS failure with cumulative rules
        if registered:
            agents._set_active_model(cheap_model)
            console.print(f"  Test: re-running {cheap_model} on {task_id} with cumulative patch rules...")
            rerun_out = f"_nway_patch_rerun_{cheap_model.replace('/', '_')}.json"
            rerun_tasks = _run_pipeline_with_patch_rules(
                cheap_model=cheap_model,
                data_dir=str(DEFAULT_DATA_DIR),
                output=rerun_out,
                max_per_class=1,
                patch_rules_path=rule_engine.path,
                failure_task_ids=[task_id],
            )
            rerun_correct = any(t.get("correct") for t in rerun_tasks if t["task_id"] == task_id)
            patch_records[-1]["rerun_correct"] = rerun_correct
            if rerun_correct:
                console.print(f"  [green]✓ FIXED[/green]")
            else:
                console.print(f"  [red]✗ STILL FAILING[/red]")

    return patch_records


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KF N-way dialogic patching loop")
    p.add_argument("--cheap-model",   dest="cheap_model",   default="google/gemma-4-26b-a4b-it")
    p.add_argument("--expert-model",  dest="expert_model",  default="claude-sonnet-4-6")
    p.add_argument("--validator-model", dest="validator_model", default="",
                   help="Model for pool validation (defaults to expert-model)")
    p.add_argument("--failures-from", dest="failures_from", default="",
                   help="Load failures from an existing N-way results JSON instead of running zero-shot")
    p.add_argument("--max-per-class",          dest="max_per_class",          type=int, default=3)
    p.add_argument("--max-val-per-class",      dest="max_val_per_class",      type=int, default=4,
                   help="Validation images per class for held-out gate (default: 4, 4×7=28 total)")
    p.add_argument("--max-authoring-per-class", dest="max_authoring_per_class", type=int, default=4)
    p.add_argument("--max-confirm-per-class",  dest="max_confirm_per_class",  type=int, default=8)
    p.add_argument("--min-precision",  dest="min_precision", type=float, default=0.75)
    p.add_argument("--data-dir",       dest="data_dir",      default=str(DEFAULT_DATA_DIR))
    p.add_argument("--patch-rules",    dest="patch_rules",   default="patch_rules_nway.json")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--output",         default="patch_session_nway.json")
    p.add_argument("--skip-rerun",     dest="skip_rerun", action="store_true")
    p.add_argument("--zero-shot-only", dest="zero_shot_only", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args               = parse_args()
    _load_api_keys()
    patch_rules_path   = _HERE / args.patch_rules
    effective_validator = args.validator_model or args.expert_model

    console.rule("[bold]KF N-way Dialogic Patching Loop[/bold]")
    console.print(f"  Cheap model:      [cyan]{args.cheap_model}[/cyan]")
    console.print(f"  Expert model:     [cyan]{args.expert_model}[/cyan]")
    console.print(f"  Validator model:  [cyan]{effective_validator}[/cyan]")
    console.print(f"  Patch rules file: [cyan]{patch_rules_path}[/cyan]")
    console.print(f"  Val pool/class:   {args.max_val_per_class} (×{len(ALL_CLASSES)} classes = "
                  f"{args.max_val_per_class * len(ALL_CLASSES)} images)")
    console.print(f"  Dry-run:          {args.dry_run}")

    console.print(f"\n[dim]Loading HAM10000 from {args.data_dir}...[/dim]")
    ds = load_ham10000(args.data_dir)

    # Step 1: Get failures
    if args.failures_from:
        console.print(f"\nLoading failures from [cyan]{args.failures_from}[/cyan]...")
        with open(args.failures_from) as f:
            prev = json.load(f)
        all_tasks = prev.get("tasks", [])
    else:
        console.print(f"\n[bold]Step 1[/bold]: Running {args.cheap_model} zero-shot N-way baseline...")
        zs_output = f"patch_nway_zeroshot_{args.cheap_model.replace('/', '_').replace('-', '_')}.json"
        all_tasks = _run_zero_shot_baseline(
            args.cheap_model, args.data_dir, zs_output, args.max_per_class
        )

    failures    = [t for t in all_tasks if not t.get("correct")]
    total       = len(all_tasks)
    before_correct = total - len(failures)

    console.print(f"\nBaseline: {before_correct}/{total} correct | "
                  f"[red]{len(failures)} failure(s) to patch[/red]")

    # Per-class failure breakdown
    class_failures: dict[str, int] = {}
    for t in all_tasks:
        lbl = t.get("correct_label", "?")
        if not t.get("correct"):
            class_failures[lbl] = class_failures.get(lbl, 0) + 1
    if class_failures:
        console.print("  Failures by class: " +
                      " | ".join(f"{k}: {v}" for k, v in sorted(class_failures.items())))

    if args.zero_shot_only:
        out_path = _HERE / args.output
        with open(out_path, "w") as f:
            json.dump({"tasks": all_tasks, "total": total, "failures": len(failures)}, f, indent=2)
        console.print(f"Zero-shot baseline saved to [cyan]{args.output}[/cyan]")
        return

    if not failures:
        console.print("[green]No failures — nothing to patch.[/green]")
        return

    # Step 2: Initialise isolated N-way patch rules
    console.print(f"\n[bold]Step 2[/bold]: Initialising N-way patch rules file...")
    rule_engine = _init_patch_rules(patch_rules_path)
    console.print(f"  Created empty: {patch_rules_path.name}")

    # Steps 3–6: Per-failure patch loop
    console.print(f"\n[bold]Steps 3–6[/bold]: Per-failure N-way patch loop...\n")

    patch_records = await run_patch_loop(
        failures=failures,
        ds=ds,
        cheap_model=args.cheap_model,
        expert_model=args.expert_model,
        rule_engine=rule_engine,
        max_val_per_class=args.max_val_per_class,
        max_authoring_per_class=args.max_authoring_per_class,
        max_confirm_per_class=args.max_confirm_per_class,
        min_precision=args.min_precision,
        dry_run=args.dry_run,
        validator_model=args.validator_model,
    )

    # Step 7: Full re-run with patch rules
    after_correct = before_correct
    if not args.skip_rerun and not args.dry_run and any(r.get("rule_registered") for r in patch_records):
        console.print(f"\n[bold]Step 7[/bold]: Re-running {args.cheap_model} on all failures with patch rules...")
        rerun_out   = f"patch_nway_rerun_{args.cheap_model.replace('/', '_').replace('-', '_')}.json"
        rerun_tasks = _run_pipeline_with_patch_rules(
            cheap_model=args.cheap_model,
            data_dir=args.data_dir,
            output=rerun_out,
            max_per_class=args.max_per_class,
            patch_rules_path=patch_rules_path,
            failure_task_ids=[t["task_id"] for t in failures],
        )
        newly_correct = sum(1 for t in rerun_tasks if t.get("correct"))
        after_correct = before_correct + newly_correct
    else:
        newly_correct = sum(
            1 for r in patch_records
            if r.get("rule_registered") and r.get("rerun_correct")
        )
        after_correct = before_correct + newly_correct

    # Summary
    rules_authored    = sum(1 for r in patch_records if r.get("rule_authored"))
    rules_accepted    = sum(1 for r in patch_records if r.get("rule_accepted"))
    rules_registered  = sum(1 for r in patch_records if r.get("rule_registered"))

    console.rule("[bold]Patch Summary[/bold]")
    tbl = Table(show_header=True)
    tbl.add_column("Phase"); tbl.add_column("Correct"); tbl.add_column("Accuracy")
    tbl.add_row("Before patching (zero-shot)",
                f"{before_correct}/{total}", f"{before_correct/max(total,1)*100:.1f}%")
    tbl.add_row("After patching (KF on failures only)",
                f"{after_correct}/{total}",  f"{after_correct/max(total,1)*100:.1f}%")
    tbl.add_row("Delta", f"+{after_correct - before_correct}",
                f"+{(after_correct - before_correct)/max(total,1)*100:.1f}pp")
    tbl.add_row("Failures patched",
                f"{newly_correct}/{len(failures)}", "")
    tbl.add_row("Rules authored/accepted/registered",
                f"{rules_authored}/{rules_accepted}/{rules_registered}", "")
    console.print(tbl)

    # Save session
    session = {
        "cheap_model":         args.cheap_model,
        "expert_model":        args.expert_model,
        "validator_model":     effective_validator,
        "patch_rules_file":    str(patch_rules_path),
        "category_set_id":     CATEGORY_SET_ID,
        "dry_run":             args.dry_run,
        "total_tasks":         total,
        "failures_before":     len(failures),
        "failures_processed":  len(patch_records),
        "rules_authored":      rules_authored,
        "rules_accepted":      rules_accepted,
        "rules_registered":    rules_registered,
        "patch_records":       patch_records,
    }
    out_path = _HERE / args.output
    with open(out_path, "w") as f:
        json.dump(session, f, indent=2)
    console.print(f"\nSession saved to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
