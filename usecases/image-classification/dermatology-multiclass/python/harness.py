"""
harness.py — CLI test runner for the KF dermatology-multiclass N-way ensemble.

Usage:
  python harness.py                                        # all 7 classes, first 3/class
  python harness.py --all                                  # all 7 classes, all test images
  python harness.py --class Melanoma                       # single class test images only
  python harness.py --all --max-per-class 5                # up to 5 test images per class
  python harness.py --all --limit 30                       # first 30 images total
  python harness.py --all --mode test                      # read-only (no learning)
  python harness.py --all --baseline zero_shot             # zero-shot baseline (no KF pipeline)
  python harness.py --all --resume                         # continue an interrupted run
  python harness.py --task-list ids.json                   # explicit list of task_ids

Data dir default: C:\\_backup\\ml\\data\\DermaMNIST_HAM10000  (override: --data-dir)
Rules file:       rules_nway.json                             (override: --rules)
Output:           results_nway.json                           (override: --output)
"""

from __future__ import annotations
import argparse
import asyncio
import datetime
import io
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_HERE    = Path(__file__).parent
_KF_ROOT = _HERE.resolve().parents[4]
_DERM2_PY = _HERE.parents[1] / "dermatology" / "python"
for _p in (str(_KF_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# _DERM2_PY appended (lowest priority) — provides rules.py and tools.py
# without shadowing the multiclass dataset.py.
if str(_DERM2_PY) not in sys.path:
    sys.path.append(str(_DERM2_PY))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import ensemble as _ensemble_mod
from ensemble import run_ensemble, DATASET_TAG
from rules import RuleEngine
from tools import ToolRegistry
from dataset import (
    load as load_ham10000,
    DEFAULT_DATA_DIR,
    HAM10000Dataset,
    ALL_CLASSES,
    CATEGORY_SET_ID,
    CATEGORY_NAMES,
    DX_TO_NAME,
    NAME_TO_DX,
)
import agents

console = Console()

DEFAULT_RULES_PATH  = str(_HERE / "rules_nway.json")
DEFAULT_OUTPUT_PATH = "results_nway.json"
DEFAULT_N_FEW_SHOT  = 2   # 2 per class for verifier (N classes × 2 = 14 images max)


# ---------------------------------------------------------------------------
# Task builder — N-way
# ---------------------------------------------------------------------------

def build_tasks(
    ds: HAM10000Dataset,
    n_few_shot: int = DEFAULT_N_FEW_SHOT,
    max_per_class: int | None = None,
    filter_class: str = "",
) -> list[dict]:
    """Build N-way task dicts for all test images.

    Each task covers all 7 lesion classes simultaneously.
    few_shot is a dict {class_name: [path, ...]} with n_few_shot images/class.

    Args:
        ds:             Loaded HAM10000Dataset.
        n_few_shot:     How many train images per class for the verifier.
        max_per_class:  Cap on test images per class (None = all).
        filter_class:   If set, only build tasks for this class name.
    """
    # Pre-build few-shot paths (fixed for all tasks in this run)
    few_shot: dict[str, list[str]] = {}
    for c in ALL_CLASSES:
        imgs = ds.sample_images(c["dx"], n=n_few_shot, split="train", seed=42)
        few_shot[c["name"]] = [str(img.file_path) for img in imgs]

    tasks: list[dict] = []
    for c in ALL_CLASSES:
        if filter_class and c["name"] != filter_class:
            continue
        test_imgs = ds.images_for_class(c["dx"], split="test")
        if max_per_class is not None:
            test_imgs = test_imgs[:max_per_class]
        for img in test_imgs:
            tasks.append({
                "category_set_id": CATEGORY_SET_ID,
                "categories":      CATEGORY_NAMES,
                "dx_codes":        [c["dx"] for c in ALL_CLASSES],
                "test_image_path": str(img.file_path),
                "test_label":      c["name"],
                "few_shot":        few_shot,
                "_image_id":       img.image_id,
                "_dx":             c["dx"],
            })
    return tasks


def task_id_for(task: dict) -> str:
    return f"{task['category_set_id']}_{task['_dx']}_{task['_image_id']}"


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_results(
    output_path: Path,
    all_results: list[dict],
    model: str,
    dataset: str,
    rules: RuleEngine,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total    = len(all_results)
    correct  = sum(1 for r in all_results if r.get("correct"))
    accuracy = correct / total if total else 0.0

    # Per-class breakdown
    class_stats: dict[str, dict] = {}
    for r in all_results:
        lbl = r.get("correct_label", "unknown")
        s   = class_stats.setdefault(lbl, {"correct": 0, "total": 0})
        s["total"] += 1
        if r.get("correct"):
            s["correct"] += 1
    for s in class_stats.values():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0.0

    # Confusion: predicted_label -> correct_label counts
    confusion: dict[str, dict[str, int]] = {}
    for r in all_results:
        pred  = r.get("predicted_label", "?")
        truth = r.get("correct_label", "?")
        confusion.setdefault(truth, {}).setdefault(pred, 0)
        confusion[truth][pred] += 1

    total_cost  = sum(r.get("cost_usd", 0.0) for r in all_results)
    avg_ms      = sum(r.get("duration_ms", 0) for r in all_results) / max(total, 1)
    total_in    = sum(r.get("input_tokens", 0) for r in all_results)
    total_out   = sum(r.get("output_tokens", 0) for r in all_results)
    total_cc    = sum(r.get("cache_creation_tokens", 0) for r in all_results)
    total_cr    = sum(r.get("cache_read_tokens", 0) for r in all_results)
    total_calls = sum(r.get("api_calls", 0) for r in all_results)
    run_ts      = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "summary": {
            "correct":              correct,
            "total":                total,
            "accuracy":             accuracy,
            "avg_ms":               avg_ms,
            "total_cost_usd":       round(total_cost, 6),
            "avg_cost_usd":         round(total_cost / max(total, 1), 6),
            "total_input_tokens":   total_in,
            "total_output_tokens":  total_out,
            "total_cache_creation": total_cc,
            "total_cache_read":     total_cr,
            "total_api_calls":      total_calls,
            "model":                model,
            "dataset":              dataset,
            "category_set_id":      CATEGORY_SET_ID,
            "timestamp":            run_ts,
            "rules":                rules.stats_summary(),
        },
        "per_class":  class_stats,
        "confusion":  confusion,
        "tasks":      all_results,
    }

    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KF Dermatology Multiclass (N-way) Ensemble Harness")
    p.add_argument("--data-dir",    default=str(DEFAULT_DATA_DIR))
    p.add_argument("--all",         action="store_true",
                   help="Run all 7 classes (default: first 3 images per class)")
    p.add_argument("--class",       dest="filter_class", default="",
                   help="Run only this class (e.g. 'Melanoma')")
    p.add_argument("--task-list",   dest="task_list", default="",
                   help="JSON list of task_ids to run")
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--max-per-class", dest="max_per_class", type=int, default=3)
    p.add_argument("--n-few-shot",  dest="n_few_shot", type=int, default=DEFAULT_N_FEW_SHOT,
                   help=f"Reference images per class for verifier (default: {DEFAULT_N_FEW_SHOT})")
    p.add_argument("--output",      default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--mode",        choices=["train", "test"], default="train")
    p.add_argument("--baseline",    choices=["zero_shot", "few_shot"], default="",
                   help="Run baseline (zero_shot or few_shot) instead of full KF pipeline")
    p.add_argument("--max-revisions", dest="max_revisions", type=int, default=None)
    p.add_argument("--dataset",     default="derm-ham10000")
    p.add_argument("--dataset-tag", dest="dataset_tag", default=DATASET_TAG)
    p.add_argument("--rules",       default=DEFAULT_RULES_PATH,
                   help="Path to N-way rules JSON file")
    p.add_argument("--model",       default="claude-sonnet-4-6")
    p.add_argument("--quiet",       action="store_true")
    p.add_argument("--prompts",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args      = parse_args()
    verbose   = not args.quiet
    test_mode = args.mode == "test"

    agents._set_show_prompts(args.prompts)
    agents._set_active_model(args.model)
    agents._set_default_model(args.model)

    if args.max_revisions is not None:
        _ensemble_mod.MAX_REVISIONS = args.max_revisions

    # API keys
    key_file = Path("P:/_access/Security/api_keys.env")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY") \
                   and not os.environ.get(k):
                    os.environ[k] = v

    # Rule engine (N-way, separate from 2-way rules.json)
    rules = RuleEngine(args.rules, dataset_tag=args.dataset_tag)

    # Tool registry (schema cache)
    tool_reg = ToolRegistry(read_only=test_mode, dataset_tag=args.dataset_tag)

    # Dataset
    console.print(f"[dim]Loading HAM10000 from {args.data_dir}...[/dim]")
    try:
        ds = load_ham10000(args.data_dir)
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        sys.exit(1)

    # Determine max_per_class: --all means all test images; default caps at 3
    max_pc = None if args.all else args.max_per_class

    all_tasks = build_tasks(
        ds,
        n_few_shot=args.n_few_shot,
        max_per_class=max_pc,
        filter_class=args.filter_class,
    )

    if args.task_list:
        tl_path = Path(args.task_list)
        if not tl_path.exists():
            console.print(f"[red]--task-list file not found: {tl_path}[/red]")
            sys.exit(1)
        id_set    = set(json.loads(tl_path.read_text(encoding="utf-8")))
        all_tasks = [t for t in all_tasks if task_id_for(t) in id_set]
        console.print(f"[dim]  --task-list: {len(all_tasks)} task(s) selected[/dim]")

    output_path  = Path(args.output)
    all_results: list[dict] = []
    completed_ids: set[str] = set()
    correct_count = 0

    if args.resume and output_path.exists():
        existing      = json.loads(output_path.read_text(encoding="utf-8"))
        all_results   = existing.get("tasks", [])
        completed_ids = {r["task_id"] for r in all_results}
        correct_count = sum(1 for r in all_results if r.get("correct"))
        console.print(f"[dim]  Resuming: {len(completed_ids)} tasks done ({correct_count} correct)[/dim]")
        all_tasks = [t for t in all_tasks if task_id_for(t) not in completed_ids]

    if args.limit is not None:
        all_tasks = all_tasks[:args.limit]

    total_tasks = len(all_tasks)
    if total_tasks == 0:
        console.print("[yellow]No tasks to run.[/yellow]")
        return

    _scope = args.filter_class or "all 7 classes"
    _mode_label = "[red]test[/red] (read-only)" if test_mode else "[green]train[/green] (learning)"
    console.print(Panel(
        f"[bold]KF Dermatology Multiclass — N-way (HAM10000)[/bold]\n"
        f"Model:       [cyan]{args.model}[/cyan]\n"
        f"Tasks:       {total_tasks}  (scope={_scope})\n"
        f"Mode:        {_mode_label}\n"
        + (f"Baseline:    {args.baseline}\n" if args.baseline else "")
        + f"Few-shot:    {args.n_few_shot} images/class\n"
        f"Categories:  {len(CATEGORY_NAMES)} — {', '.join(CATEGORY_NAMES)}\n"
        f"NS tag:      [cyan]{args.dataset_tag}[/cyan]\n"
        f"Rules:       {args.rules}  {rules.stats_summary()}\n"
        f"Output:      {args.output}",
        title="Harness"
    ))

    run_results: list[dict] = []

    for i, task in enumerate(all_tasks, 1):
        tid = task_id_for(task)
        console.rule(f"[{i}/{total_tasks}] {tid}")

        if args.baseline:
            bl_decision, bl_ms = await agents.run_baseline(task, mode=args.baseline)
            bl_label       = bl_decision.get("label", "uncertain")
            bl_correct     = bl_label == task["test_label"]
            prev_cost      = sum(r.get("cost_usd", 0) for r in all_results + run_results)
            meta = {
                "task_id":         tid,
                "category_set_id": task["category_set_id"],
                "predicted_label": bl_label,
                "correct_label":   task["test_label"],
                "correct":         bl_correct,
                "confidence":      bl_decision.get("confidence", 0.0),
                "reasoning":       bl_decision.get("reasoning", ""),
                "duration_ms":     bl_ms,
                "cost_usd":        agents.get_cost_tracker().cost_usd() - prev_cost,
                "api_calls":       1,
                "model":           args.model,
                "baseline_mode":   args.baseline,
                "_dx":             task.get("_dx", ""),
            }
        else:
            meta = await run_ensemble(
                task=task,
                task_id=tid,
                rule_engine=rules,
                tool_registry=tool_reg,
                verbose=verbose,
                dataset=args.dataset,
                dataset_tag=args.dataset_tag,
                max_revisions=_ensemble_mod.MAX_REVISIONS,
                test_mode=test_mode,
            )
            if meta.get("correct"):
                correct_count += 1

        all_results.append(meta)
        run_results.append(meta)
        _save_results(output_path, all_results, meta.get("model", "?"), args.dataset, rules)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    rs          = run_results
    total       = len(rs)
    n_correct   = sum(1 for r in rs if r.get("correct"))
    accuracy    = n_correct / total if total else 0.0
    avg_ms      = sum(r.get("duration_ms", 0) for r in rs) / max(total, 1)
    total_cost  = sum(r.get("cost_usd", 0.0) for r in rs)
    total_in    = sum(r.get("input_tokens", 0) for r in rs)
    total_out   = sum(r.get("output_tokens", 0) for r in rs)
    total_cc    = sum(r.get("cache_creation_tokens", 0) for r in rs)
    total_cr    = sum(r.get("cache_read_tokens", 0) for r in rs)
    total_calls = sum(r.get("api_calls", 0) for r in rs)

    t = Table(title="Run Summary (this invocation)")
    t.add_column("Metric"); t.add_column("Value")
    t.add_row("Tasks run",        str(total))
    t.add_row("Correct",          f"{n_correct}/{total}  ({accuracy*100:.1f}%)")
    t.add_row("Avg duration",     f"{avg_ms/1000:.1f}s")
    t.add_row("Total cost (USD)", f"${total_cost:.4f}")
    t.add_row("Avg cost / task",  f"${total_cost/max(total,1):.4f}")
    t.add_row("Input tokens",     f"{total_in:,}")
    t.add_row("Output tokens",    f"{total_out:,}")
    t.add_row("Cache create",     f"{total_cc:,}")
    t.add_row("Cache read",       f"{total_cr:,}")
    t.add_row("Total API calls",  f"{total_calls:,}")
    t.add_row("Model",            args.model)
    t.add_row("Rules (active)",   str(rules.stats_summary()["active"]))
    console.print(t)

    # Per-class accuracy table
    class_acc: dict[str, dict] = {}
    for r in rs:
        lbl = r.get("correct_label", "?")
        s   = class_acc.setdefault(lbl, {"correct": 0, "total": 0})
        s["total"] += 1
        if r.get("correct"):
            s["correct"] += 1

    pt = Table(title="Per-Class Accuracy")
    pt.add_column("Class"); pt.add_column("Correct"); pt.add_column("Total"); pt.add_column("Accuracy")
    for cls_name in CATEGORY_NAMES:
        if cls_name in class_acc:
            s   = class_acc[cls_name]
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0
            pt.add_row(cls_name, str(s["correct"]), str(s["total"]), f"{pct:.1f}%")
    console.print(pt)

    console.print(f"Results written to [bold]{args.output}[/bold]")


if __name__ == "__main__":
    asyncio.run(main())
