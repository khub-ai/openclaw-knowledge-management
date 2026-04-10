"""
test_dialogic_rules_expanded.py

Test the 3 dialogic-distilled rules against all 27 melanoma failures
from the expanded baseline (30/class). Uses Sonnet as validator —
no OpenRouter needed.

For each failure image, checks whether ANY of the 3 rules fires.
If a rule fires on a melanoma → the patched classifier would get it right.

Usage:
  python test_dialogic_rules_expanded.py
"""
import asyncio, json, os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_ham10000
from rich.console import Console
from rich.table import Table

console = Console()

DATA_DIR = r"C:\_backup\ml\data\DermaMNIST_HAM10000"
VALIDATOR = "claude-sonnet-4-6"


def _load_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    for line in kf.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip(); v = v.strip()
            if k == "ANTHROPIC_API_KEY" and not os.environ.get(k):
                os.environ[k] = v


async def main():
    _load_keys()

    # Load rules
    with open(_HERE / "patch_rules_dialogic.json", encoding="utf-8") as f:
        rules_data = json.load(f)
    rules = rules_data["rules"]
    console.print(f"Loaded [bold]{len(rules)}[/bold] dialogic rules")

    # Convert to validator format
    for r in rules:
        r["rule"] = r["rule_text"]
        r["favors"] = "Melanoma"

    # Load baseline failures
    with open(_HERE / "expanded_baseline_qwen_mel_nev.json", encoding="utf-8") as f:
        baseline = json.load(f)
    all_tasks = baseline["tasks"]
    failures = [t for t in all_tasks if not t["correct"]]
    console.print(f"Baseline: {len(all_tasks) - len(failures)}/{len(all_tasks)} correct, "
                  f"{len(failures)} failures")

    # Build image_id → path map
    ds = load_ham10000(DATA_DIR)
    all_mel = (ds.sample_images("mel", 500, split="test", seed=0)
               + ds.sample_images("mel", 500, split="train", seed=0))
    img_map = {img.image_id: str(img.file_path) for img in all_mel}

    # Test each rule on each failure
    results = []
    fixed_count = 0

    for i, task in enumerate(failures, 1):
        tid = task["task_id"]
        iid = tid.split("melanoma_vs_melanocytic_nevus_")[1]
        path = img_map.get(iid)
        if not path:
            console.print(f"[red]Not found: {iid}[/red]")
            continue

        fired_rules = []
        for rule in rules:
            val_result, _ = await agents.run_rule_validator_on_image(
                image_path=path,
                ground_truth="Melanoma",
                candidate_rule=rule,
                model=VALIDATOR,
            )
            if val_result.get("precondition_met", False):
                fired_rules.append(rule["id"])

        fixed = len(fired_rules) > 0
        if fixed:
            fixed_count += 1
        status = "[green]FIXED[/green]" if fixed else "[red]still wrong[/red]"
        rules_str = ", ".join(fired_rules) if fired_rules else "none"

        console.print(f"  [{i:2d}/{len(failures)}] {iid}  {status}  rules={rules_str}")
        results.append({
            "image_id": iid,
            "fired_rules": fired_rules,
            "fixed": fixed,
        })

    # Summary
    console.print()
    before = len(all_tasks) - len(failures)
    after = before + fixed_count
    total = len(all_tasks)

    tbl = Table(title="Expanded Test — Dialogic Rules", show_header=True)
    tbl.add_column("Phase"); tbl.add_column("Correct"); tbl.add_column("Accuracy")
    tbl.add_row("Zero-shot baseline", f"{before}/{total}", f"{before/total*100:.1f}%")
    tbl.add_row("After dialogic rules", f"{after}/{total}", f"{after/total*100:.1f}%")
    tbl.add_row("Delta", f"+{fixed_count}", f"+{fixed_count/total*100:.1f}pp")
    console.print(tbl)

    # Per-rule coverage
    console.print("\nPer-rule fire count on failures:")
    for rule in rules:
        fires = sum(1 for r in results if rule["id"] in r["fired_rules"])
        console.print(f"  {rule['id']} ({rule['triggered_by']}): "
                      f"fires on {fires}/{len(failures)} failures")

    # Save
    out = {
        "baseline_correct": before,
        "baseline_total": total,
        "failures_tested": len(failures),
        "fixed_by_rules": fixed_count,
        "after_correct": after,
        "after_accuracy": round(after / total, 4),
        "delta_pp": round(fixed_count / total * 100, 1),
        "per_image": results,
    }
    with open(_HERE / "dialogic_expanded_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    console.print(f"\nSaved to dialogic_expanded_results.json")


if __name__ == "__main__":
    asyncio.run(main())
