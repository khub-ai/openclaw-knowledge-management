"""
create_benchmark.py — Generate fixed benchmark manifests for road surface DD experiments.

Run once by the maintainer (requires local RSCD zip).
Commit the resulting JSON files to git.
Other developers use load_benchmark() to reproduce the same image set.

Three manifest types:
  probe     — 12 images per class for PUPIL Domain Readiness Probe
  pool      — 20 images per class for DD pool validation
  failures  — N images where a specified PUPIL model makes wrong predictions

Usage:
  python create_benchmark.py --pair dry_vs_wet --types probe,pool
  python create_benchmark.py --pair dry_vs_wet --types probe,pool --annotate-difficulty
  python create_benchmark.py --pair dry_vs_wet --types failures \\
      --pupil-model qwen/qwen3-vl-8b-instruct --n-failures 8
  python create_benchmark.py --pair dry_vs_wet --types probe,pool,failures \\
      --annotate-difficulty --pupil-model qwen/qwen3-vl-8b-instruct
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KF_ROOT = _HERE.parents[4]
if str(_KF_ROOT) not in sys.path:
    sys.path.insert(0, str(_KF_ROOT))

import agents
from dataset import load as load_rscd, CONFUSABLE_PAIRS, DEFAULT_DATA_DIR
from domain_config import ROAD_SURFACE_CONFIG
from benchmark import (
    save_manifest, BENCHMARK_TYPE_PROBE,
    BENCHMARK_TYPE_POOL, BENCHMARK_TYPE_FAILURES,
    DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD,
)

from rich.console import Console
from rich.progress import track

console = Console()

_BENCHMARKS_DIR = _HERE.parent / "benchmarks"

_TMP_DIR = (_HERE / ".." / ".." / ".." / ".." / ".tmp" / "rscd_session").resolve()

# Seeds — different per manifest type so images don't overlap
_SEED_PROBE = 7
_SEED_POOL  = 42   # same seed as original distill_dialogic.py pool


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def _load_api_keys():
    kf = Path("P:/_access/Security/api_keys.env")
    if kf.exists():
        for line in kf.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY") and not os.environ.get(k):
                    os.environ[k] = v


# ---------------------------------------------------------------------------
# Difficulty annotation via TUTOR
# ---------------------------------------------------------------------------

async def _annotate_difficulty(
    images_with_paths: list,   # [(RoadImage, path_str), ...]
    pair_info: dict,
    tutor_model: str,
    call_agent_fn,
) -> dict:
    """Ask TUTOR to rate visual ambiguity (easy/medium/hard) for each image.

    Returns dict: image_id -> difficulty string.
    """
    from core.dialogic_distillation.agents import image_block, parse_json_block

    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]
    difficulties = {}

    for road_img, path in track(images_with_paths, description="Annotating difficulty..."):
        system = (
            f"You are a {ROAD_SURFACE_CONFIG.expert_role}. "
            f"Rate the visual difficulty of classifying this road surface image."
        )
        content = [
            image_block(path),
            {"type": "text", "text": (
                f"This image belongs to class: {road_img.friction_label}\n\n"
                f"Rate how visually difficult it is to distinguish this "
                f"'{road_img.friction_label}' image from '{class_b if road_img.friction_label == class_a else class_a}'.\n\n"
                f"Consider:\n"
                f"  easy  — class is unambiguous; visual cues are strong and obvious\n"
                f"  medium — class is distinguishable but requires attention\n"
                f"  hard  — subtle; could plausibly be confused with the other class\n\n"
                f"Respond with JSON: "
                f'{{\"difficulty\": \"easy\" | \"medium\" | \"hard\", '
                f'\"reason\": \"one sentence\"}}'
            )},
        ]
        raw, _ = await call_agent_fn(
            "TUTOR_ANNOTATE", content,
            system_prompt=system,
            model=tutor_model,
            max_tokens=128,
        )
        result = (parse_json_block(raw) or {})
        diff = result.get("difficulty", DIFFICULTY_MEDIUM).lower().strip()
        if diff not in (DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD):
            diff = DIFFICULTY_MEDIUM
        difficulties[road_img.image_id] = diff

    return difficulties


# ---------------------------------------------------------------------------
# Failure discovery
# ---------------------------------------------------------------------------

async def _discover_failures(
    ds,
    pair_info: dict,
    pupil_model: str,
    n_per_class: int,
    n_failures: int,
    call_agent_fn,
) -> list:
    """Run PUPIL on test images and return failure records."""
    from core.dialogic_distillation.agents import image_block, parse_json_block

    friction_a  = pair_info["friction_a"]
    friction_b  = pair_info["friction_b"]
    class_a     = pair_info["class_a"]
    class_b     = pair_info["class_b"]
    mat_filter  = pair_info.get("material_filter")

    images_a = ds.sample_images(friction_a, n_per_class, split="test",
                                seed=_SEED_PROBE, material_filter=mat_filter)
    images_b = ds.sample_images(friction_b, n_per_class, split="test",
                                seed=_SEED_PROBE, material_filter=mat_filter)
    all_images = images_a + images_b

    console.print(f"  Running PUPIL on {len(all_images)} test images...")
    failures = []

    for img in track(all_images, description="Discovering failures..."):
        if len(failures) >= n_failures:
            break

        true_label  = class_a if img.friction == friction_a else class_b
        path        = str(img.resolve_path(_TMP_DIR))

        content = [
            image_block(path),
            {"type": "text", "text": (
                f"Classify this road surface as one of:\n"
                f"  A) {class_a}\n  B) {class_b}\n\n"
                f"Respond with JSON: "
                f'{{\"prediction\": \"{class_a}\" or \"{class_b}\", '
                f'\"reasoning\": \"brief\"}}'
            )},
        ]
        raw, _ = await call_agent_fn(
            "PUPIL", content,
            system_prompt="You are a vision model classifying road surface conditions.",
            model=pupil_model,
            max_tokens=256,
        )
        from core.dialogic_distillation.agents import parse_json_block
        result = parse_json_block(raw) or {}
        pred   = result.get("prediction", "").strip()

        if class_a.lower() in pred.lower():
            pred = class_a
        elif class_b.lower() in pred.lower():
            pred = class_b
        else:
            pred = class_a if "A" in pred else class_b

        if pred != true_label:
            failures.append({
                "road_img":        img,
                "correct_label":   true_label,
                "wrong_prediction": pred,
                "pupil_reasoning": result.get("reasoning", raw[:200]),
            })

    return failures[:n_failures]


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------

def _image_entry(road_img, difficulty: str = DIFFICULTY_MEDIUM, notes: str = "") -> dict:
    """Build a single manifest image entry from a RoadImage."""
    return {
        "image_id":   road_img.image_id,
        "filename":   f"{road_img.image_id}-{road_img.friction}-"
                      f"{road_img.material}-{road_img.roughness}.jpg",
        "friction":   road_img.friction,
        "material":   road_img.material,
        "roughness":  road_img.roughness,
        "true_class": (
            next(
                (pair["class_a"] if road_img.friction == pair["friction_a"]
                 else pair["class_b"])
                for pair in CONFUSABLE_PAIRS
                if road_img.friction in (pair["friction_a"], pair["friction_b"])
            )
        ),
        "difficulty": difficulty,
        "notes":      notes,
    }


async def build_probe_manifest(
    ds,
    pair_info: dict,
    n_per_class: int,
    tutor_model: str,
    annotate: bool,
    call_agent_fn,
) -> dict:
    mat_filter = pair_info.get("material_filter")
    friction_a = pair_info["friction_a"]
    friction_b = pair_info["friction_b"]
    class_a    = pair_info["class_a"]
    class_b    = pair_info["class_b"]

    imgs_a = ds.sample_images(friction_a, n_per_class, split="train",
                              seed=_SEED_PROBE, material_filter=mat_filter)
    imgs_b = ds.sample_images(friction_b, n_per_class, split="train",
                              seed=_SEED_PROBE, material_filter=mat_filter)

    console.print(f"  Probe: sampled {len(imgs_a)} {class_a} + {len(imgs_b)} {class_b}")

    difficulty_map: dict = {}
    if annotate:
        console.print("  Annotating difficulty (calls TUTOR)...")
        pairs = (
            [(img, str(img.resolve_path(_TMP_DIR))) for img in imgs_a]
            + [(img, str(img.resolve_path(_TMP_DIR))) for img in imgs_b]
        )
        difficulty_map = await _annotate_difficulty(pairs, pair_info, tutor_model, call_agent_fn)
    else:
        # Assign difficulty by roughness as a structural proxy (revisable later)
        _roughness_difficulty = {"smooth": DIFFICULTY_EASY, "slight": DIFFICULTY_MEDIUM, "severe": DIFFICULTY_HARD}
        for img in imgs_a + imgs_b:
            difficulty_map[img.image_id] = _roughness_difficulty.get(img.roughness, DIFFICULTY_MEDIUM)

    images = [
        _image_entry(img, difficulty=difficulty_map.get(img.image_id, DIFFICULTY_MEDIUM))
        for img in imgs_a + imgs_b
    ]

    return {
        "benchmark_id":    f"road_surface_{pair_info['pair_id']}_probe_v1",
        "version":         "1.0.0",
        "benchmark_type":  BENCHMARK_TYPE_PROBE,
        "created":         str(date.today()),
        "pair_id":         pair_info["pair_id"],
        "class_a":         class_a,
        "class_b":         class_b,
        "material_filter": mat_filter,
        "rscd_split":      "train",
        "selection_seed":  _SEED_PROBE,
        "pupil_model":     None,
        "n_per_class":     n_per_class,
        "description": (
            f"Fixed probe image set for PUPIL Domain Readiness Probe. "
            f"{len(imgs_a)} {class_a} + {len(imgs_b)} {class_b} images, "
            f"sampled from RSCD training split (seed={_SEED_PROBE})."
        ),
        "images":          images,
    }


async def build_pool_manifest(
    ds,
    pair_info: dict,
    n_per_class: int,
    call_agent_fn,
) -> dict:
    mat_filter = pair_info.get("material_filter")
    friction_a = pair_info["friction_a"]
    friction_b = pair_info["friction_b"]
    class_a    = pair_info["class_a"]
    class_b    = pair_info["class_b"]

    imgs_a = ds.sample_images(friction_a, n_per_class, split="train",
                              seed=_SEED_POOL, material_filter=mat_filter)
    imgs_b = ds.sample_images(friction_b, n_per_class, split="train",
                              seed=_SEED_POOL, material_filter=mat_filter)

    console.print(f"  Pool: sampled {len(imgs_a)} {class_a} + {len(imgs_b)} {class_b}")

    images = [_image_entry(img) for img in imgs_a + imgs_b]

    return {
        "benchmark_id":    f"road_surface_{pair_info['pair_id']}_pool_v1",
        "version":         "1.0.0",
        "benchmark_type":  BENCHMARK_TYPE_POOL,
        "created":         str(date.today()),
        "pair_id":         pair_info["pair_id"],
        "class_a":         class_a,
        "class_b":         class_b,
        "material_filter": mat_filter,
        "rscd_split":      "train",
        "selection_seed":  _SEED_POOL,
        "pupil_model":     None,
        "n_per_class":     n_per_class,
        "description": (
            f"Fixed pool validation images for DD experiments. "
            f"{len(imgs_a)} {class_a} + {len(imgs_b)} {class_b}, "
            f"sampled from RSCD training split (seed={_SEED_POOL}). "
            f"Matches seed used in original distill_dialogic.py experiment."
        ),
        "images":          images,
    }


async def build_failures_manifest(
    ds,
    pair_info: dict,
    pupil_model: str,
    n_failures: int,
    call_agent_fn,
) -> dict:
    class_a = pair_info["class_a"]
    class_b = pair_info["class_b"]

    console.print(f"  Discovering {n_failures} failures for {pupil_model}...")
    failures = await _discover_failures(
        ds, pair_info, pupil_model,
        n_per_class=40, n_failures=n_failures,
        call_agent_fn=call_agent_fn,
    )

    console.print(f"  Found {len(failures)} failures")

    # Normalise pupil model name for use in benchmark_id
    model_tag = pupil_model.replace("/", "_").replace("-", "_").replace(".", "_")

    images = []
    for f in failures:
        entry = _image_entry(f["road_img"])
        entry["notes"] = (
            f"PUPIL predicted '{f['wrong_prediction']}'; "
            f"correct is '{f['correct_label']}'. "
            f"Reasoning: {f['pupil_reasoning'][:120]}"
        )
        images.append(entry)

    return {
        "benchmark_id":    f"road_surface_{pair_info['pair_id']}_failures_{model_tag}_v1",
        "version":         "1.0.0",
        "benchmark_type":  BENCHMARK_TYPE_FAILURES,
        "created":         str(date.today()),
        "pair_id":         pair_info["pair_id"],
        "class_a":         class_a,
        "class_b":         class_b,
        "material_filter": pair_info.get("material_filter"),
        "rscd_split":      "test",
        "selection_seed":  _SEED_PROBE,
        "pupil_model":     pupil_model,
        "n_per_class":     40,
        "description": (
            f"Images where {pupil_model} made wrong predictions on the "
            f"{pair_info['pair_id']} confusable pair. "
            f"Used as trigger cases for DD sessions."
        ),
        "images":          images,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate fixed benchmark manifests for road surface DD experiments")
    p.add_argument("--pair", default="dry_vs_wet",
                   choices=[cp["pair_id"] for cp in CONFUSABLE_PAIRS])
    p.add_argument("--types", default="probe,pool",
                   help="Comma-separated list: probe, pool, failures")
    p.add_argument("--n-probe",    type=int, default=12,
                   help="Images per class for probe manifest")
    p.add_argument("--n-pool",     type=int, default=20,
                   help="Images per class for pool manifest")
    p.add_argument("--n-failures", type=int, default=8,
                   help="Number of failure cases to discover")
    p.add_argument("--pupil-model", default="qwen/qwen3-vl-8b-instruct",
                   help="PUPIL model for failure discovery")
    p.add_argument("--tutor-model", default="claude-opus-4-6",
                   help="TUTOR model for difficulty annotation")
    p.add_argument("--annotate-difficulty", action="store_true",
                   help="Call TUTOR to rate visual difficulty of each image")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", default=str(_BENCHMARKS_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be generated without calling APIs or writing files")
    return p.parse_args()


async def main():
    args = parse_args()
    _load_api_keys()

    types = [t.strip() for t in args.types.split(",")]

    console.rule("[bold]Road Surface Benchmark Creator[/bold]")
    console.print(f"  Pair:    [cyan]{args.pair}[/cyan]")
    console.print(f"  Types:   [cyan]{', '.join(types)}[/cyan]")
    console.print(f"  Output:  [cyan]{args.output_dir}[/cyan]")
    if args.dry_run:
        console.print("  [yellow]DRY RUN — no files will be written[/yellow]")

    pair_info = next(cp for cp in CONFUSABLE_PAIRS if cp["pair_id"] == args.pair)

    # Load dataset
    console.print(f"\n[dim]Loading RSCD from {args.data_dir}...[/dim]")
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_rscd(args.data_dir)
    except FileNotFoundError as e:
        console.print(f"\n[red]Dataset not found:[/red] {e}")
        console.print(
            "\n[yellow]Download RSCD:[/yellow]\n"
            "  kaggle datasets download cristvollerei/rscd-dataset-1million\n"
            "  Place zip at C:\\\\backup\\\\ml\\\\data\\\\rscd-dataset-1million.zip"
        )
        return

    stats = ds.class_stats()
    total = sum(sum(s.values()) for s in stats.values())
    console.print(f"  Loaded: {total:,} images\n")

    agents.ACTIVE_MODEL = args.tutor_model
    call_fn = agents.call_agent
    output_dir = Path(args.output_dir)

    manifests_written = []

    # Probe
    if BENCHMARK_TYPE_PROBE in types:
        console.print("[bold]Building probe manifest...[/bold]")
        manifest = await build_probe_manifest(
            ds, pair_info, args.n_probe,
            args.tutor_model, args.annotate_difficulty, call_fn,
        )
        out = output_dir / f"{pair_info['pair_id']}_probe_v1.json"
        if not args.dry_run:
            save_manifest(manifest, out)
            console.print(f"  Saved: [cyan]{out.name}[/cyan] ({len(manifest['images'])} images)")
        else:
            console.print(f"  Would save: [cyan]{out.name}[/cyan] ({len(manifest['images'])} images)")
        manifests_written.append(out.name)

    # Pool
    if BENCHMARK_TYPE_POOL in types:
        console.print("[bold]Building pool manifest...[/bold]")
        manifest = await build_pool_manifest(ds, pair_info, args.n_pool, call_fn)
        out = output_dir / f"{pair_info['pair_id']}_pool_v1.json"
        if not args.dry_run:
            save_manifest(manifest, out)
            console.print(f"  Saved: [cyan]{out.name}[/cyan] ({len(manifest['images'])} images)")
        else:
            console.print(f"  Would save: [cyan]{out.name}[/cyan] ({len(manifest['images'])} images)")
        manifests_written.append(out.name)

    # Failures
    if BENCHMARK_TYPE_FAILURES in types:
        console.print(f"[bold]Building failures manifest for {args.pupil_model}...[/bold]")
        agents.ACTIVE_MODEL = args.pupil_model
        manifest = await build_failures_manifest(
            ds, pair_info, args.pupil_model, args.n_failures, call_fn,
        )
        model_tag = args.pupil_model.replace("/", "_").replace("-", "_").replace(".", "_")
        out = output_dir / f"{pair_info['pair_id']}_failures_{model_tag}_v1.json"
        if not args.dry_run:
            save_manifest(manifest, out)
            console.print(f"  Saved: [cyan]{out.name}[/cyan] ({len(manifest['images'])} failures)")
        else:
            console.print(f"  Would save: [cyan]{out.name}[/cyan] ({len(manifest['images'])} failures)")
        manifests_written.append(out.name)

    console.print(f"\n[green]Done.[/green]  Files to commit:")
    for name in manifests_written:
        console.print(f"  git add usecases/image-classification/road-surface/benchmarks/{name}")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())
