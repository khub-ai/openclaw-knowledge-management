# Road Surface Benchmark Image Sets

Fixed, versioned sets of RSCD image IDs used for:

1. **Probe benchmarks** — PUPIL Domain Readiness Probe (model-agnostic)
2. **DD pool benchmarks** — pool validation images for DD experiments (model-agnostic)
3. **Failure benchmarks** — known PUPIL failure cases for DD sessions (per-model)

---

## Why manifests rather than raw images

RSCD contains ~600K images (~XX GB). Committing images to git is not feasible.
Instead this directory contains **manifest JSON files** — lists of image IDs,
metadata, and difficulty annotations — that are used to extract the relevant
images from the user's local RSCD zip on demand.

Other developers or independent benchmarking runs can reproduce the exact same
image set by pointing `load_benchmark()` at their local RSCD zip.

---

## Files

```
benchmarks/
  dry_vs_wet_probe_v1.json        24 images (12 dry + 12 wet) — probe
  dry_vs_wet_pool_v1.json         40 images (20 dry + 20 wet) — DD pool
  dry_vs_wet_failures_qwen3_v1.json   8 images — Qwen3-VL-8B failures
  wet_vs_water_probe_v1.json      (future)
```

All manifests follow the schema described below.

---

## Manifest schema

```json
{
  "benchmark_id":    "road_surface_dry_vs_wet_probe_v1",
  "version":         "1.0.0",
  "benchmark_type":  "probe" | "pool" | "failures",
  "created":         "YYYY-MM-DD",
  "pair_id":         "dry_vs_wet",
  "class_a":         "Dry Road",
  "class_b":         "Wet Road",
  "material_filter": "asphalt",
  "rscd_split":      "train" | "test" | "val",
  "selection_seed":  42,
  "pupil_model":     null,
  "description":     "...",
  "images": [
    {
      "image_id":   "2022012523413511",
      "filename":   "2022012523413511-dry-asphalt-smooth.jpg",
      "friction":   "dry",
      "material":   "asphalt",
      "roughness":  "smooth",
      "true_class": "Dry Road",
      "difficulty": "easy" | "medium" | "hard",
      "notes":      ""
    },
    ...
  ]
}
```

`pupil_model` is null for model-agnostic manifests and set for failure manifests.
`difficulty` is assigned by the TUTOR during `create_benchmark.py --annotate-difficulty`.

---

## Generating manifests (run once by maintainer)

Requires: local RSCD zip at `C:\_backup\ml\data\rscd-dataset-1million.zip`
(or `--data-dir` pointing to another location).

```bash
cd usecases/image-classification/road-surface/python

# Generate probe + pool manifests (deterministic, no API calls)
python create_benchmark.py --pair dry_vs_wet --types probe,pool

# Also annotate difficulty levels (calls TUTOR — uses API)
python create_benchmark.py --pair dry_vs_wet --types probe,pool --annotate-difficulty

# Generate failure manifest for Qwen3-VL-8B (calls PUPIL — uses API)
python create_benchmark.py --pair dry_vs_wet --types failures \
    --pupil-model qwen/qwen3-vl-8b-instruct --n-failures 8

# Full regeneration of all manifests
python create_benchmark.py --pair dry_vs_wet --types probe,pool,failures \
    --annotate-difficulty --pupil-model qwen/qwen3-vl-8b-instruct
```

Commit the resulting JSON files to git. Do not commit any extracted image files.

---

## Using manifests in code

```python
from dataset import load as load_rscd
from benchmark import load_benchmark, to_probe_images

ds = load_rscd("C:/_backup/ml/data")
tmp_dir = Path(".tmp/rscd_session")

# Load probe images — returns list of (RoadImage, ProbeImage) pairs
items = load_benchmark("benchmarks/dry_vs_wet_probe_v1.json", ds, tmp_dir)

# Or get just ProbeImage list for use with probe()
probe_images = to_probe_images("benchmarks/dry_vs_wet_probe_v1.json", ds, tmp_dir)
```

---

## Versioning policy

- Increment version (v1 → v2) when images are added, removed, or reannotated.
- Keep older versions — published results reference specific version IDs.
- Never modify a committed manifest's `images` list in place; create a new version.
