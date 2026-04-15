"""
dataset.py — HAM10000 dataset loader for the KF dermatology-multiclass use case.

N-way: all 7 HAM10000 lesion classes, no confusable-pair framing.

Provides:
  load(data_dir)        Load HAM10000 from a directory.
  HAM10000Dataset       Container with per-class split access and sampling.
  DermImage             Single image record.
  ALL_CLASSES           Ordered list of {dx, name} dicts for all 7 classes.
  CATEGORY_SET_ID       Canonical identifier for the 7-class category set.
  CATEGORY_NAMES        Ordered list of human-readable class names.
  DX_CODES              Ordered list of dx codes (same order as CATEGORY_NAMES).
  DX_TO_NAME            Dict: dx code -> human-readable name.
  NAME_TO_DX            Dict: human-readable name -> dx code.

Split logic (unchanged from 2-way):
  Per-dx: sort unique lesion_ids, 80% train / 20% test.
  Test set: one image per test lesion_id (first alphabetically).
  Train set: all images from train lesion_ids.
"""

from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_DATA_DIR = Path(r"C:\_backup\ml\data\DermaMNIST_HAM10000")

_SEED = 42

# ---------------------------------------------------------------------------
# 7-class category set
# ---------------------------------------------------------------------------

ALL_CLASSES: list[dict] = [
    {"dx": "mel",   "name": "Melanoma"},
    {"dx": "nv",    "name": "Melanocytic Nevus"},
    {"dx": "bkl",   "name": "Benign Keratosis"},
    {"dx": "bcc",   "name": "Basal Cell Carcinoma"},
    {"dx": "akiec", "name": "Actinic Keratosis"},
    {"dx": "vasc",  "name": "Vascular Lesion"},
    {"dx": "df",    "name": "Dermatofibroma"},
]

CATEGORY_SET_ID: str                   = "dermatology_7class"
CATEGORY_NAMES:  list[str]             = [c["name"] for c in ALL_CLASSES]
DX_CODES:        list[str]             = [c["dx"]   for c in ALL_CLASSES]
DX_TO_NAME:      dict[str, str]        = {c["dx"]:   c["name"] for c in ALL_CLASSES}
NAME_TO_DX:      dict[str, str]        = {c["name"]: c["dx"]   for c in ALL_CLASSES}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DermImage:
    image_id:  str
    lesion_id: str
    dx:        str
    file_path: Path
    dx_type:   str


@dataclass
class HAM10000Dataset:
    _images: List[DermImage]
    _split_index: Dict[str, Dict[str, List[DermImage]]]

    def images_for_class(self, dx: str, split: str = "test") -> List[DermImage]:
        """Return all images for a dx code in the given split."""
        return self._split_index.get(dx, {}).get(split, [])

    def sample_images(
        self,
        dx: str,
        n: int,
        split: str = "train",
        seed: int = _SEED,
    ) -> List[DermImage]:
        """Return up to n images, one per unique lesion_id, from the given split."""
        imgs = self.images_for_class(dx, split=split)
        seen: set[str] = set()
        unique: List[DermImage] = []
        for img in sorted(imgs, key=lambda x: x.image_id):
            if img.lesion_id not in seen:
                seen.add(img.lesion_id)
                unique.append(img)
        rng = random.Random(seed)
        return rng.sample(unique, min(n, len(unique)))

    def class_name(self, dx: str) -> str:
        return DX_TO_NAME.get(dx, dx)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _find_image_path(data_dir: Path, image_id: str) -> Optional[Path]:
    fname = f"{image_id}.jpg"
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        p = data_dir / part / fname
        if p.exists():
            return p
    return None


def load(data_dir: Optional[str | Path] = None) -> HAM10000Dataset:
    """Load HAM10000 from disk and build train/test splits per dx class."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    metadata_path = root / "HAM10000_metadata.csv"

    rows: List[dict] = []
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    dx_lesion_images: Dict[str, Dict[str, List[tuple]]] = {}
    for row in rows:
        dx        = row["dx"].strip()
        lesion_id = row["lesion_id"].strip()
        image_id  = row["image_id"].strip()
        dx_type   = row.get("dx_type", "").strip()
        dx_lesion_images.setdefault(dx, {}).setdefault(lesion_id, []).append((image_id, dx_type))

    split_index: Dict[str, Dict[str, List[DermImage]]] = {}
    all_images: List[DermImage] = []

    for dx, lesion_map in dx_lesion_images.items():
        sorted_lesion_ids = sorted(lesion_map.keys())
        n          = len(sorted_lesion_ids)
        split_pt   = int(n * 0.8)
        train_lesions = set(sorted_lesion_ids[:split_pt])
        test_lesions  = set(sorted_lesion_ids[split_pt:])

        train_imgs: List[DermImage] = []
        test_imgs:  List[DermImage] = []

        for lesion_id, entries in lesion_map.items():
            sorted_entries = sorted(entries, key=lambda e: e[0])
            if lesion_id in train_lesions:
                for iid, dxt in sorted_entries:
                    fp = _find_image_path(root, iid) or (root / "HAM10000_images_part_1" / f"{iid}.jpg")
                    img = DermImage(image_id=iid, lesion_id=lesion_id, dx=dx, file_path=fp, dx_type=dxt)
                    train_imgs.append(img); all_images.append(img)
            elif lesion_id in test_lesions:
                iid, dxt = sorted_entries[0]
                fp = _find_image_path(root, iid) or (root / "HAM10000_images_part_1" / f"{iid}.jpg")
                img = DermImage(image_id=iid, lesion_id=lesion_id, dx=dx, file_path=fp, dx_type=dxt)
                test_imgs.append(img); all_images.append(img)

        train_imgs.sort(key=lambda x: x.image_id)
        test_imgs.sort(key=lambda x: x.image_id)
        split_index[dx] = {"train": train_imgs, "test": test_imgs}

    return HAM10000Dataset(_images=all_images, _split_index=split_index)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = load()
    print(f"HAM10000 loaded — {CATEGORY_SET_ID} ({len(ALL_CLASSES)} classes)\n")
    for c in ALL_CLASSES:
        train = ds.images_for_class(c["dx"], split="train")
        test  = ds.images_for_class(c["dx"], split="test")
        print(f"  {c['dx']:6s} ({c['name']:25s})  train={len(train):4d}  test={len(test):4d}")
