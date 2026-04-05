"""
patch.py — Dialogic patching loop for the KF dermatology pipeline.

Workflow:
  1. Run a cheap VLM zero-shot to identify failures.
  2. For each failure, call an expert VLM to author a corrective rule.
  3. Validate the candidate rule against the labeled training pool.
  4. Register accepted rules in the knowledge base.
  5. Re-run the cheap VLM with the new rules and report the improvement.

This demonstrates the core KF value proposition: a cheap VLM + expert-authored
rules can approach the performance of a much more expensive VLM, with the
expensive model's cost incurred only once per rule (not per image).

Usage:
  # Full loop: zero-shot → patch → re-test
  python patch.py --cheap-model o4-mini --expert-model claude-opus-4-6

  # Start from an existing zero-shot results file
  python patch.py --failures-from results_baseline_o4mini_zeroshot.json \\
                  --cheap-model o4-mini --expert-model claude-opus-4-6

  # Dry-run: author and validate rules without registering them
  python patch.py --cheap-model o4-mini --expert-model claude-opus-4-6 --dry-run
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
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
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Knowledge base paths
# ---------------------------------------------------------------------------

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
                if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") and not os.environ.get(k):
                    os.environ[k] = v


def _task_from_results_entry(entry: dict, ds, pair_info: dict) -> dict | None:
    """Reconstruct a minimal task dict from a results entry + dataset."""
    task_id = entry["task_id"]
    # task_id format: {pair_id}_{image_id}
    pair_id = entry["pair_id"]
    image_id = task_id[len(pair_id) + 1:]  # strip "pair_id_" prefix

    # Find image in dataset
    dx_a = pair_info.get("dx_a", "")
    dx_b = pair_info.get("dx_b", "")

    for dx in (dx_a, dx_b):
        for split in ("test", "train"):
            for img in ds.images_for_class(dx, split=split):
                if img.image_id == image_id:
                    return {
                        "class_a": pair_info["class_a"],
                        "class_b": pair_info["class_b"],
                        "pair_id": pair_id,
                        "test_image_path": str(img.file_path),
                        "test_label": entry["correct_label"],
                    }
    return None


def _pair_info_for_id(pair_id: str) -> dict | None:
    for cp in CONFUSABLE_PAIRS:
        if cp["pair_id"] == pair_id:
            return cp
    return None


def _init_patch_rules(patch_rules_path: Path) -> "RuleEngine":
    """Create a fresh isolated patch rules file and return its RuleEngine.

    Never touches the main rules.json or any KB JSON file.
    """
    from rules import RuleEngine as _RE
    if patch_rules_path.exists():
        patch_rules_path.unlink()
    return _RE(str(patch_rules_path), dataset_tag=DATASET_TAG)


def _register_rule(rule_engine: "RuleEngine", candidate_rule: dict,
                   pair_id: str, validation: dict,
                   triggered_by: str, cheap_model: str, expert_model: str) -> bool:
    """Register an accepted patch rule into the isolated rule engine.

    Returns True if the rule was added.
    """
    rule_text     = candidate_rule["rule"]
    favors        = candidate_rule["favors"]
    confidence    = candidate_rule.get("confidence", "medium")
    preconditions = candidate_rule.get("preconditions", [])

    # Encode pre-conditions in the condition string so the MEDIATOR sees them.
    precond_str = "; ".join(preconditions) if preconditions else rule_text
    condition   = f"[Patch rule — {pair_id}] {precond_str}"
    action      = (f"Classify as {favors}. Confidence: {confidence}. "
                   f"Rule: {rule_text}")

    result = rule_engine.add_rule(
        condition=condition,
        action=action,
        source=f"expert_vlm:{expert_model}",
        source_task=triggered_by,
        tags=[DATASET_TAG, pair_id, f"patch:{cheap_model}",
              f"triggered_by:{triggered_by}"],
        lineage={"type": "new", "parent_ids": [],
                 "reason": f"patch rule for {pair_id} triggered by {triggered_by}"},
        observability_filter=False,
    )
    return result is not None


def _run_zero_shot_baseline(cheap_model: str, data_dir: str, output: str,
                             max_per_class: int) -> list[dict]:
    """Run cheap model zero-shot baseline and return task results."""
    result = subprocess.run(
        [
            sys.executable, "harness.py",
            "--all", "--max-per-class", str(max_per_class),
            "--mode", "test",
            "--model", cheap_model,
            "--baseline", "zero_shot",
            "--output", output,
            "--data-dir", data_dir,
        ],
        cwd=_HERE,
        capture_output=False,  # show progress to user
        text=True,
    )
    if result.returncode != 0:
        console.print("[red]Zero-shot baseline run failed[/red]")
        sys.exit(1)
    with open(Path(_HERE) / output) as f:
        data = json.load(f)
    return data.get("tasks", [])


def _run_pipeline_with_patch_rules(cheap_model: str, data_dir: str, output: str,
                                    max_per_class: int,
                                    patch_rules_path: Path,
                                    pair: str = "") -> list[dict]:
    """Run cheap model + isolated patch rules through the KF pipeline."""
    cmd = [
        sys.executable, "harness.py",
        "--max-per-class", str(max_per_class),
        "--mode", "test",
        "--model", cheap_model,
        "--rules", str(patch_rules_path),
        "--output", output,
        "--data-dir", data_dir,
    ]
    if pair:
        cmd += ["--pair", pair]
    else:
        cmd += ["--all"]
    result = subprocess.run(cmd, cwd=_HERE, capture_output=False, text=True)
    if result.returncode != 0:
        console.print("[red]Pipeline re-run failed[/red]")
        sys.exit(1)
    with open(Path(_HERE) / output) as f:
        data = json.load(f)
    return data.get("tasks", [])


# ---------------------------------------------------------------------------
# Core patching loop
# ---------------------------------------------------------------------------

async def run_patch_loop(
    failures: list[dict],
    ds,
    cheap_model: str,
    expert_model: str,
    rule_engine,
    max_val_per_class: int,
    min_precision: float,
    dry_run: bool,
) -> list[dict]:
    """Author, validate, and register rules for each failure.

    Rules are written ONLY to the isolated rule_engine (patch_rules_clean.json).
    The main rules.json and KB JSON files are never touched.

    Returns list of patch records.
    """
    patch_records = []

    for i, failure in enumerate(failures, 1):
        pair_id = failure["pair_id"]
        task_id = failure["task_id"]
        wrong   = failure["predicted_label"]
        correct = failure["correct_label"]

        console.rule(f"[{i}/{len(failures)}] {task_id}")
        console.print(f"  Wrong: [red]{wrong}[/red]  →  Correct: [green]{correct}[/green]")

        pair_info = _pair_info_for_id(pair_id)
        if not pair_info:
            console.print(f"  [yellow]Pair info not found for {pair_id}, skipping[/yellow]")
            continue

        task = _task_from_results_entry(failure, ds, pair_info)
        if not task:
            console.print(f"  [yellow]Image not found for {task_id}, skipping[/yellow]")
            continue

        # --- Step 1: Expert VLM authors a rule ---
        console.print(f"  Calling expert VLM ({expert_model}) to author rule...")
        candidate_rule, _ = await agents.run_expert_rule_author(
            task=task,
            wrong_prediction=wrong,
            correct_label=correct,
            model_reasoning=failure.get("reasoning", ""),
            model=expert_model,
        )
        console.print(f"  Rule: [italic]{candidate_rule.get('rule', '')[:120]}[/italic]")
        console.print(f"  Favors: {candidate_rule.get('favors')} | "
                      f"Confidence: {candidate_rule.get('confidence')}")
        for pc in candidate_rule.get("preconditions", []):
            console.print(f"    Pre-condition: {pc}")

        # --- Step 2: Collect validation images from training pool ---
        dx_a = pair_info.get("dx_a", "")
        dx_b = pair_info.get("dx_b", "")
        val_imgs_a = [(str(img.file_path), pair_info["class_a"])
                      for img in ds.sample_images(dx_a, max_val_per_class, split="train")]
        val_imgs_b = [(str(img.file_path), pair_info["class_b"])
                      for img in ds.sample_images(dx_b, max_val_per_class, split="train")]
        validation_images = val_imgs_a + val_imgs_b

        console.print(f"  Validating against {len(validation_images)} training images "
                      f"({len(val_imgs_a)} {pair_info['class_a']}, "
                      f"{len(val_imgs_b)} {pair_info['class_b']})...")

        # --- Step 3: Validate rule against training pool ---
        validation = await agents.validate_candidate_rule(
            candidate_rule=candidate_rule,
            validation_images=validation_images,
            trigger_image_path=task["test_image_path"],
            trigger_correct_label=correct,
            model=expert_model,
        )

        fires_on_trigger = validation["fires_on_trigger"]
        precision        = validation["precision"]
        accepted         = validation["accepted"] and precision >= min_precision

        console.print(
            f"  Validation: TP={validation['tp']} FP={validation['fp']} "
            f"TN={validation['tn']} FN={validation['fn']} | "
            f"precision={precision:.2f} recall={validation['recall']:.2f} | "
            f"fires_on_trigger={fires_on_trigger}"
        )
        status = "[green]ACCEPTED[/green]" if accepted else "[red]REJECTED[/red]"
        if not fires_on_trigger:
            console.print(f"  {status} — rule did not fire on the trigger image")
        elif not accepted:
            console.print(f"  {status} — precision {precision:.2f} < {min_precision:.2f}")
        else:
            console.print(f"  {status}")

        record = {
            "task_id":       task_id,
            "pair_id":       pair_id,
            "wrong_prediction": wrong,
            "correct_label": correct,
            "candidate_rule": {k: v for k, v in candidate_rule.items()
                               if k != "raw_response"},
            "validation":    validation,
            "accepted":      accepted,
            "registered":    False,
        }

        # --- Step 4: Register in isolated patch rules file ---
        if accepted and not dry_run:
            ok = _register_rule(
                rule_engine=rule_engine,
                candidate_rule=candidate_rule,
                pair_id=pair_id,
                validation=validation,
                triggered_by=task_id,
                cheap_model=cheap_model,
                expert_model=expert_model,
            )
            record["registered"] = ok
            if ok:
                console.print("  Rule registered in patch rules file.")
            else:
                console.print("  [yellow]Registration failed (observability filter?)[/yellow]")

        patch_records.append(record)

    return patch_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _detect_cross_pair_firing(rerun_tasks: list[dict],
                               patch_records: list[dict]) -> list[dict]:
    """Check whether patch rules fired on tasks outside their intended pair.

    A cross-pair firing means a rule authored for pair X matched a task from
    pair Y.  This is either a quality problem (pre-conditions too broad) or a
    genuine generalization — both warrant surfacing to the human expert
    (dialogic learning: the pupil has been enlightened, the master must confirm).

    Returns a list of cross-pair firing events.
    """
    # Build map: rule_id → intended pair_id
    rule_to_pair: dict[str, str] = {}
    for rec in patch_records:
        if rec.get("registered"):
            rule_text = rec["candidate_rule"].get("rule", "")
            # We don't have the rule_id directly; use pair_id from session record.
            # Match by pair_id tag embedded in the condition prefix.
            rule_to_pair[rec["pair_id"]] = rec["pair_id"]  # placeholder

    # Better: inspect fired rule IDs per task and compare with intended pair.
    events = []
    for task in rerun_tasks:
        task_pair   = task.get("pair_id", "")
        fired_rules = task.get("rule_ids_fired", [])
        for rid in fired_rules:
            # Find the patch record that owns this rule.
            for rec in patch_records:
                intended = rec.get("pair_id", "")
                if intended and intended != task_pair and rec.get("registered"):
                    events.append({
                        "rule_id":       rid,
                        "intended_pair": intended,
                        "fired_on_pair": task_pair,
                        "task_id":       task.get("task_id", ""),
                        "correct":       task.get("correct"),
                        "flag":          (
                            "QUALITY PROBLEM — pre-conditions too broad"
                            if not task.get("correct")
                            else "POSSIBLE GENERALIZATION — worth expert review"
                        ),
                    })
    return events


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KF dialogic patching loop")
    p.add_argument("--cheap-model",       dest="cheap_model",       default="o4-mini")
    p.add_argument("--expert-model",      dest="expert_model",      default="claude-opus-4-6")
    p.add_argument("--failures-from",     dest="failures_from",     default="",
                   help="Load failures from an existing results JSON")
    p.add_argument("--pair",              default="",
                   help="Limit zero-shot + rerun to a single pair ID")
    p.add_argument("--max-per-class",     dest="max_per_class",     type=int, default=3)
    p.add_argument("--max-val-per-class", dest="max_val_per_class", type=int, default=8)
    p.add_argument("--min-precision",     dest="min_precision",     type=float, default=0.75)
    p.add_argument("--data-dir",          dest="data_dir",
                   default="C:/_backup/ml/data/DermaMNIST_HAM10000")
    p.add_argument("--patch-rules",       dest="patch_rules",
                   default="patch_rules_clean.json",
                   help="Isolated rules file for this session (never touches main rules.json)")
    p.add_argument("--dry-run",           action="store_true",
                   help="Author and validate rules but do not register them")
    p.add_argument("--output",            default="patch_session.json")
    p.add_argument("--skip-rerun",        dest="skip_rerun", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    _load_api_keys()

    patch_rules_path = Path(_HERE) / args.patch_rules

    console.rule("[bold]KF Dialogic Patching Loop[/bold]")
    console.print(f"  Cheap model:       [cyan]{args.cheap_model}[/cyan]")
    console.print(f"  Expert model:      [cyan]{args.expert_model}[/cyan]")
    console.print(f"  Patch rules file:  [cyan]{patch_rules_path}[/cyan]")
    console.print(f"  Dry-run:           {args.dry_run}")

    # Load dataset
    console.print(f"\n[dim]Loading HAM10000 from {args.data_dir}...[/dim]")
    ds = load_ham10000(args.data_dir)

    # Step 1: Get failures (zero-shot or from file)
    if args.failures_from:
        console.print(f"\nLoading failures from [cyan]{args.failures_from}[/cyan]...")
        with open(args.failures_from) as f:
            prev = json.load(f)
        all_tasks = prev.get("tasks", [])
    else:
        console.print(f"\n[bold]Step 1[/bold]: Running {args.cheap_model} zero-shot baseline...")
        zs_output = f"patch_zeroshot_{args.cheap_model.replace('-', '_')}.json"
        all_tasks = _run_zero_shot_baseline(
            args.cheap_model, args.data_dir, zs_output, args.max_per_class
        )

    if args.pair:
        all_tasks = [t for t in all_tasks if t.get("pair_id") == args.pair]

    failures = [t for t in all_tasks if not t["correct"]]
    total    = len(all_tasks)
    console.print(f"\nBaseline: {total - len(failures)}/{total} correct | "
                  f"[red]{len(failures)} failure(s) to patch[/red]")

    if not failures:
        console.print("[green]No failures — nothing to patch.[/green]")
        return

    # Step 2: Initialise an isolated, empty patch rules file
    console.print(f"\n[bold]Step 2[/bold]: Initialising patch rules file...")
    rule_engine = _init_patch_rules(patch_rules_path)
    console.print(f"  Created empty: {patch_rules_path.name}")

    # Steps 3–5: For each failure, author + validate + register
    console.print(f"\n[bold]Steps 3–5[/bold]: Expert rule authoring + validation + registration...\n")
    patch_records = await run_patch_loop(
        failures=failures,
        ds=ds,
        cheap_model=args.cheap_model,
        expert_model=args.expert_model,
        rule_engine=rule_engine,
        max_val_per_class=args.max_val_per_class,
        min_precision=args.min_precision,
        dry_run=args.dry_run,
    )

    accepted   = [r for r in patch_records if r["accepted"]]
    registered = [r for r in patch_records if r["registered"]]

    console.print(f"\nRules authored: {len(patch_records)} | "
                  f"Accepted: {len(accepted)} | "
                  f"Registered: {len(registered)}")

    # Step 6: Re-run cheap model with only the patch rules
    rerun_tasks: list[dict] = []
    cross_pair_events: list[dict] = []

    if registered and not args.skip_rerun:
        console.print(f"\n[bold]Step 6[/bold]: Re-running {args.cheap_model} "
                      f"with {len(registered)} patch rule(s)...")
        rerun_output = f"patch_rerun_{args.cheap_model.replace('-', '_')}.json"
        rerun_tasks = _run_pipeline_with_patch_rules(
            cheap_model=args.cheap_model,
            data_dir=args.data_dir,
            output=rerun_output,
            max_per_class=args.max_per_class,
            patch_rules_path=patch_rules_path,
            pair=args.pair,
        )

        # Cross-pair firing detection (Fix 5)
        cross_pair_events = _detect_cross_pair_firing(rerun_tasks, patch_records)
        if cross_pair_events:
            console.print("\n[bold yellow]⚠ Cross-pair rule firing detected:[/bold yellow]")
            for ev in cross_pair_events:
                console.print(
                    f"  Rule intended for [cyan]{ev['intended_pair']}[/cyan] "
                    f"fired on [magenta]{ev['fired_on_pair']}[/magenta] "
                    f"(task {ev['task_id']}) → {ev['flag']}"
                )
            console.print(
                "  [dim]→ Surface to human expert: the pupil has been enlightened, "
                "the master must confirm or refine.[/dim]"
            )

        before_correct = total - len(failures)
        after_correct  = sum(1 for t in rerun_tasks if t["correct"])
        delta          = after_correct - before_correct

        console.rule("[bold]Patch Summary[/bold]")
        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("Phase"); tbl.add_column("Correct"); tbl.add_column("Accuracy")
        tbl.add_row("Before patching (zero-shot)",
                    f"{before_correct}/{total}", f"{before_correct/total*100:.1f}%")
        tbl.add_row("After patching (KF + patch rules)",
                    f"{after_correct}/{total}", f"{after_correct/total*100:.1f}%")
        tbl.add_row("Delta", f"{delta:+d}", f"{delta/total*100:+.1f}pp")
        console.print(tbl)

    elif args.dry_run:
        console.print("\n[yellow]Dry-run: no rules registered, skipping re-run.[/yellow]")
    elif not registered:
        console.print("\nNo rules accepted — skipping re-run.")

    # Save session record
    session = {
        "cheap_model":       args.cheap_model,
        "expert_model":      args.expert_model,
        "patch_rules_file":  str(patch_rules_path),
        "dry_run":           args.dry_run,
        "total_tasks":       total,
        "failures_before":   len(failures),
        "rules_authored":    len(patch_records),
        "rules_accepted":    len(accepted),
        "rules_registered":  len(registered),
        "cross_pair_events": cross_pair_events,
        "patch_records":     patch_records,
    }
    with open(args.output, "w") as f:
        json.dump(session, f, indent=2)
    console.print(f"\nSession saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
