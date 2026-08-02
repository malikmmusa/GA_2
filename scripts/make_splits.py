#!/usr/bin/env python3
"""
Build a frozen, eye-grouped train/test split for the disc + fovea models.

Why this exists
---------------
The previous split was ``df.sample(frac=0.8, random_state=42)`` over filenames.
Because ``<id>.png`` and ``<id>_2.png`` are the *same eye* at two timepoints,
that split put 9 of 10 validation images in the same eye as a training image.
Validation was scoring the model on eyes it had already memorised.

This script groups by eye, so both timepoints of an eye always land on the same
side of the split, and writes the assignment to disk **once**. Re-running is a
no-op unless ``--force`` is passed, so metrics stay comparable across runs even
as new cases are added.

Structure of the output:
  - ``test``  : frozen holdout eyes. Never train on these, never early-stop on
                these. Touch them only when reporting a final number.
  - ``folds`` : the remaining eyes split into K grouped CV folds, for model
                selection and early stopping.

Usage:
  python scripts/make_splits.py                       # create (refuses to clobber)
  python scripts/make_splits.py --verify              # check split vs current labels
  python scripts/make_splits.py --force               # rebuild from scratch
  python scripts/make_splits.py --assign-new          # place unseen eyes into folds
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CSV = PROJECT_ROOT / "data" / "training" / "disc_labels_v2.csv"
DEFAULT_OUT = PROJECT_ROOT / "data" / "splits" / "splits_v1.json"

SPLIT_VERSION = 1
DEFAULT_SEED = 20260802
DEFAULT_TEST_FRACTION = 0.25
DEFAULT_N_FOLDS = 5

# ``22028370.png`` and ``22028370_2.png`` are the same eye at two timepoints.
_TIMEPOINT_SUFFIX = re.compile(r"_2$")


def eye_id(filename: str) -> str:
    """Map an image filename to the eye it belongs to.

    ``23242134.png`` and ``23242134_2.png`` both map to ``23242134``.
    """
    return _TIMEPOINT_SUFFIX.sub("", Path(filename).stem)


def group_images_by_eye(filenames: List[str]) -> Dict[str, List[str]]:
    """Return ``{eye_id: [filenames...]}`` with deterministic ordering."""
    groups: Dict[str, List[str]] = {}
    for fname in sorted(filenames):
        groups.setdefault(eye_id(fname), []).append(fname)
    return groups


def build_split(
    groups: Dict[str, List[str]],
    seed: int,
    test_fraction: float,
    n_folds: int,
) -> Dict:
    """Partition eyes into a frozen test set plus K grouped CV folds."""
    eyes = sorted(groups.keys())
    rng = random.Random(seed)
    shuffled = eyes[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_eyes = sorted(shuffled[:n_test])
    dev_eyes = shuffled[n_test:]

    # Round-robin over the shuffled remainder keeps folds balanced in eye count.
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for i, eye in enumerate(dev_eyes):
        folds[i % n_folds].append(eye)

    return {
        "version": SPLIT_VERSION,
        "created": date.today().isoformat(),
        "seed": seed,
        "test_fraction": test_fraction,
        "n_folds": n_folds,
        "grouping": "eye_id = image filename stem with trailing '_2' removed",
        "note": (
            "Frozen split. Do not regenerate without --force; metrics are only "
            "comparable across runs while this file is unchanged. Test eyes are "
            "holdout: never train on them and never early-stop on them."
        ),
        "n_eyes": len(eyes),
        "n_images": sum(len(v) for v in groups.values()),
        "test": {
            "eyes": test_eyes,
            "images": sorted(img for e in test_eyes for img in groups[e]),
        },
        "folds": [
            {
                "fold": i,
                "eyes": sorted(fold_eyes),
                "images": sorted(img for e in fold_eyes for img in groups[e]),
            }
            for i, fold_eyes in enumerate(folds)
        ],
    }


def split_eye_sets(split: Dict) -> Dict[str, set]:
    """Return ``{'test': {...}, 'dev': {...}}`` as eye-id sets."""
    test_eyes = set(split["test"]["eyes"])
    dev_eyes = {e for f in split["folds"] for e in f["eyes"]}
    return {"test": test_eyes, "dev": dev_eyes}


def load_split(path) -> Dict:
    """Load a frozen split JSON, with a pointed error if it hasn't been built."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen split not found: {path}\n"
            f"Create it with:  python scripts/make_splits.py"
        )
    return json.loads(path.read_text())


def split_dataframe(df: pd.DataFrame, split_path, val_fold: int):
    """Split a label dataframe into (train, val) using the frozen eye grouping.

    The holdout test eyes are excluded from *both* returns — training must never
    see them, and early stopping must never select on them. ``val_fold`` picks
    which CV fold acts as validation; the other folds are training.
    """
    split = load_split(split_path)
    n_folds = len(split["folds"])
    if not 0 <= val_fold < n_folds:
        raise ValueError(f"val_fold must be in [0, {n_folds - 1}], got {val_fold}")

    test_eyes = set(split["test"]["eyes"])
    val_eyes = set(split["folds"][val_fold]["eyes"])

    eyes = df["filename"].astype(str).map(eye_id)
    held_out = eyes.isin(test_eyes)
    is_val = eyes.isin(val_eyes)

    if held_out.any():
        print(f"Excluding {int(held_out.sum())} holdout-test image(s) from training and validation")

    unassigned = ~(held_out | is_val | eyes.isin(
        {e for i, f in enumerate(split["folds"]) if i != val_fold for e in f["eyes"]}
    ))
    if unassigned.any():
        missing = sorted(set(eyes[unassigned]))
        raise ValueError(
            f"{len(missing)} eye(s) are not in the frozen split: {missing}\n"
            f"Run:  python scripts/make_splits.py --assign-new"
        )

    return df[~held_out & ~is_val].copy(), df[is_val].copy()


def verify(split: Dict, groups: Dict[str, List[str]]) -> int:
    """Check a split against the current label set. Returns a process exit code."""
    sets = split_eye_sets(split)
    split_eyes = sets["test"] | sets["dev"]
    current_eyes = set(groups.keys())

    problems = 0

    overlap = sets["test"] & sets["dev"]
    if overlap:
        print(f"[FAIL] {len(overlap)} eye(s) appear in BOTH test and folds: {sorted(overlap)}")
        problems += 1
    else:
        print("[ok]   test and dev folds are disjoint by eye")

    # The leakage class that motivated this script: two images of one eye split apart.
    image_to_side: Dict[str, str] = {}
    for img in split["test"]["images"]:
        image_to_side[img] = "test"
    for fold in split["folds"]:
        for img in fold["images"]:
            image_to_side[img] = "dev"
    straddling = [
        eye for eye, imgs in groups.items()
        if len({image_to_side[i] for i in imgs if i in image_to_side}) > 1
    ]
    if straddling:
        print(f"[FAIL] {len(straddling)} eye(s) straddle test/dev: {sorted(straddling)}")
        problems += 1
    else:
        print("[ok]   no eye has images on both sides of the holdout boundary")

    missing = split_eyes - current_eyes
    if missing:
        print(f"[warn] {len(missing)} eye(s) in the split are no longer in the label CSV: {sorted(missing)}")

    unseen = current_eyes - split_eyes
    if unseen:
        print(f"[warn] {len(unseen)} eye(s) in the label CSV are not in the split: {sorted(unseen)}")
        print("       Run with --assign-new to add them to the CV folds (test set stays frozen).")
    else:
        print("[ok]   every labelled eye is covered by the split")

    n_test_img = len(split["test"]["images"])
    n_dev_img = sum(len(f["images"]) for f in split["folds"])
    print(
        f"\nTest: {len(sets['test'])} eyes / {n_test_img} images   "
        f"Dev: {len(sets['dev'])} eyes / {n_dev_img} images   "
        f"({split['n_folds']} folds)"
    )
    return 1 if problems else 0


def assign_new(split: Dict, groups: Dict[str, List[str]]) -> Dict:
    """Add eyes not yet in the split to the CV folds, leaving the test set frozen.

    New eyes always go to the dev folds. Growing the holdout would change what
    the test number means, which defeats the purpose of freezing it.
    """
    sets = split_eye_sets(split)
    unseen = sorted(set(groups.keys()) - (sets["test"] | sets["dev"]))
    if not unseen:
        print("No new eyes to assign.")
        return split

    # Smallest fold first, so repeated additions stay balanced.
    for eye in unseen:
        target = min(split["folds"], key=lambda f: len(f["eyes"]))
        target["eyes"] = sorted(target["eyes"] + [eye])
        target["images"] = sorted(target["images"] + groups[eye])
        print(f"  {eye} -> fold {target['fold']}")

    split["n_eyes"] = len(groups)
    split["n_images"] = sum(len(v) for v in groups.values())
    split["last_assigned"] = date.today().isoformat()
    print(f"\nAssigned {len(unseen)} new eye(s) to CV folds. Test set unchanged.")
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/verify the frozen eye-grouped split.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Label CSV with a 'filename' column")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Split JSON path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--verify", action="store_true", help="Check the existing split, write nothing")
    parser.add_argument("--assign-new", action="store_true", help="Add unseen eyes to CV folds")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"[ERROR] Label CSV not found: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    if "filename" not in df.columns:
        print(f"[ERROR] {args.csv} has no 'filename' column")
        return 1

    groups = group_images_by_eye(df["filename"].astype(str).tolist())
    print(f"Labels: {len(df)} images across {len(groups)} eyes  ({args.csv})\n")

    if args.verify:
        if not args.out.exists():
            print(f"[ERROR] No split to verify at {args.out}")
            return 1
        return verify(json.loads(args.out.read_text()), groups)

    if args.assign_new:
        if not args.out.exists():
            print(f"[ERROR] No split at {args.out}; create one first")
            return 1
        split = assign_new(json.loads(args.out.read_text()), groups)
        args.out.write_text(json.dumps(split, indent=2) + "\n")
        return 0

    if args.out.exists() and not args.force:
        print(f"[ERROR] Split already exists: {args.out}")
        print("        This file is frozen on purpose — regenerating it invalidates")
        print("        every metric measured against the old one.")
        print("        Use --verify to inspect it, --assign-new to add eyes, or")
        print("        --force if you genuinely intend to start a new baseline.")
        return 1

    split = build_split(groups, args.seed, args.test_fraction, args.folds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2) + "\n")

    print(f"Wrote {args.out}")
    print(f"  test: {len(split['test']['eyes'])} eyes / {len(split['test']['images'])} images (frozen holdout)")
    for fold in split["folds"]:
        print(f"  fold {fold['fold']}: {len(fold['eyes'])} eyes / {len(fold['images'])} images")
    print()
    return verify(split, groups)


if __name__ == "__main__":
    sys.exit(main())
