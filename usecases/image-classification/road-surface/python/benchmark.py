"""
benchmark.py — Fixed benchmark manifest loader for road surface DD experiments.

Benchmark manifests are JSON files committed to git that specify exact RSCD
image IDs for use in:
  - PUPIL Domain Readiness Probe   (benchmark_type = "probe")
  - DD pool validation             (benchmark_type = "pool")
  - Known PUPIL failure cases      (benchmark_type = "failures")

Images are not stored in git — only their IDs and metadata.  The loader
extracts images on demand from the user's local RSCD zip.

Public API:
  load_benchmark(manifest_path, ds, tmp_dir) -> BenchmarkSet
  to_probe_images(manifest_path, ds, tmp_dir) -> list[ProbeImage]
  to_pool_images(manifest_path, ds, tmp_dir)  -> list[(path, label)]
  save_manifest(manifest, path)
  load_manifest(path) -> dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_BENCHMARKS_DIR = _HERE.parent / "benchmarks"

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

BENCHMARK_TYPE_PROBE    = "probe"
BENCHMARK_TYPE_POOL     = "pool"
BENCHMARK_TYPE_FAILURES = "failures"

DIFFICULTY_EASY   = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD   = "hard"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkImageRecord:
    """One image entry in a benchmark manifest."""
    image_id:   str
    filename:   str
    friction:   str
    material:   str
    roughness:  str
    true_class: str
    difficulty: str = DIFFICULTY_MEDIUM
    notes:      str = ""


@dataclass
class BenchmarkSet:
    """Loaded benchmark — manifest metadata + resolved (RoadImage, BenchmarkImageRecord) pairs."""
    benchmark_id:    str
    version:         str
    benchmark_type:  str
    pair_id:         str
    class_a:         str
    class_b:         str
    material_filter: Optional[str]
    rscd_split:      str
    selection_seed:  int
    pupil_model:     Optional[str]
    description:     str
    records:         List[BenchmarkImageRecord] = field(default_factory=list)
    # resolved paths — populated by load_benchmark()
    resolved_paths:  List[Tuple[str, str, str]] = field(default_factory=list)
    # (image_path, true_class, difficulty)

    @property
    def n_images(self) -> int:
        return len(self.records)

    def images_for_class(self, true_class: str) -> List[Tuple[str, str, str]]:
        """Return (path, true_class, difficulty) tuples filtered by class."""
        return [(p, c, d) for p, c, d in self.resolved_paths if c == true_class]

    def as_pool_list(self) -> List[Tuple[str, str]]:
        """Return [(path, true_class), ...] for use as DD validation pool."""
        return [(p, c) for p, c, _ in self.resolved_paths]


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def save_manifest(manifest: dict, path: str | Path) -> None:
    """Write a benchmark manifest dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path: str | Path) -> dict:
    """Load a benchmark manifest dict from a JSON file."""
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Benchmark loading — resolves image IDs to filesystem paths
# ---------------------------------------------------------------------------

def load_benchmark(
    manifest_path: str | Path,
    ds,
    tmp_dir: str | Path,
) -> BenchmarkSet:
    """Load a benchmark manifest and resolve image paths from the RSCD dataset.

    Args:
        manifest_path: Path to the manifest JSON file (absolute or relative to
                       the benchmarks/ directory).
        ds:            Loaded RSCDDataset (from dataset.load()).
        tmp_dir:       Directory for extracted images. Images are extracted
                       from the zip on first access and reused thereafter.

    Returns:
        BenchmarkSet with resolved_paths populated.

    Raises:
        FileNotFoundError:  if manifest_path does not exist.
        KeyError:           if a manifest image_id is not found in the dataset.
    """
    manifest_path = _resolve_manifest_path(manifest_path)
    manifest = load_manifest(manifest_path)
    tmp_dir = Path(tmp_dir)

    # Build image_id → RoadImage lookup from full dataset
    img_map = {img.image_id: img for img in ds._images}

    records = []
    resolved = []
    missing = []

    for entry in manifest["images"]:
        rec = BenchmarkImageRecord(
            image_id   = entry["image_id"],
            filename   = entry["filename"],
            friction   = entry["friction"],
            material   = entry["material"],
            roughness  = entry["roughness"],
            true_class = entry["true_class"],
            difficulty = entry.get("difficulty", DIFFICULTY_MEDIUM),
            notes      = entry.get("notes", ""),
        )
        records.append(rec)

        road_img = img_map.get(rec.image_id)
        if road_img is None:
            missing.append(rec.image_id)
            continue

        path = str(road_img.resolve_path(tmp_dir))
        resolved.append((path, rec.true_class, rec.difficulty))

    if missing:
        raise KeyError(
            f"Benchmark '{manifest['benchmark_id']}': "
            f"{len(missing)} image(s) not found in dataset: {missing[:5]}"
            + (" ..." if len(missing) > 5 else "")
        )

    return BenchmarkSet(
        benchmark_id   = manifest["benchmark_id"],
        version        = manifest.get("version", "1.0.0"),
        benchmark_type = manifest.get("benchmark_type", BENCHMARK_TYPE_PROBE),
        pair_id        = manifest["pair_id"],
        class_a        = manifest["class_a"],
        class_b        = manifest["class_b"],
        material_filter= manifest.get("material_filter"),
        rscd_split     = manifest.get("rscd_split", "train"),
        selection_seed = manifest.get("selection_seed", 42),
        pupil_model    = manifest.get("pupil_model"),
        description    = manifest.get("description", ""),
        records        = records,
        resolved_paths = resolved,
    )


def to_probe_images(
    manifest_path: str | Path,
    ds,
    tmp_dir: str | Path,
) -> list:
    """Load a probe benchmark manifest and return a list of ProbeImage objects.

    Convenience wrapper around load_benchmark() for direct use with probe().

    Returns:
        list[ProbeImage]  — ready to pass to core.dialogic_distillation.probe.probe()
    """
    from core.dialogic_distillation.probe import ProbeImage

    bset = load_benchmark(manifest_path, ds, tmp_dir)
    return [
        ProbeImage(
            path       = path,
            true_class = true_class,
            difficulty = difficulty,
        )
        for path, true_class, difficulty in bset.resolved_paths
    ]


def to_pool_images(
    manifest_path: str | Path,
    ds,
    tmp_dir: str | Path,
) -> List[Tuple[str, str]]:
    """Load a pool benchmark manifest and return [(image_path, true_class), ...].

    Convenience wrapper for direct use as the pool_images argument to
    run_dialogic_distillation().
    """
    bset = load_benchmark(manifest_path, ds, tmp_dir)
    return bset.as_pool_list()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_manifest_path(path: str | Path) -> Path:
    """Resolve manifest path — if relative, look in benchmarks/ dir."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    candidate = _BENCHMARKS_DIR / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Manifest not found: '{path}'. "
        f"Looked in cwd and {_BENCHMARKS_DIR}."
    )


def list_benchmarks() -> List[dict]:
    """Return metadata for all manifests in the benchmarks/ directory."""
    results = []
    for json_file in sorted(_BENCHMARKS_DIR.glob("*.json")):
        try:
            m = load_manifest(json_file)
            results.append({
                "file":           json_file.name,
                "benchmark_id":   m.get("benchmark_id", "?"),
                "version":        m.get("version", "?"),
                "benchmark_type": m.get("benchmark_type", "?"),
                "pair_id":        m.get("pair_id", "?"),
                "n_images":       len(m.get("images", [])),
                "description":    m.get("description", ""),
            })
        except Exception:
            pass
    return results
