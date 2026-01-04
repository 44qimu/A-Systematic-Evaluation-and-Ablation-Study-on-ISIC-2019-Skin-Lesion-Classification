#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate indices_isic2019.pkl (5-fold train/val splits) for ISIC2019.

Output format (pickle):
{
  "trainIndCV": [np.ndarray, ...] length = n_splits
  "valIndCV":   [np.ndarray, ...] length = n_splits
}

Default paths (relative to project root = folder that contains "data/dataDir"):
- labels.csv :  data/dataDir/labels/official/labels.csv
- output pkl :  data/saveDir/indices_isic2019.pkl

This script does NOT require scikit-learn (implements stratified split manually).
"""

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    """Find the folder that contains data/dataDir."""
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "data" / "dataDir").exists():
            return p
    return start


DEFAULT_CLASS_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def infer_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Infer single-class label y for each row.
    Priority:
      1) one-hot columns (DEFAULT_CLASS_COLS)
      2) 'target' column
      3) 'label' column
    """
    class_cols = [c for c in DEFAULT_CLASS_COLS if c in df.columns]
    if len(class_cols) >= 2:
        arr = df[class_cols].to_numpy()
        y = arr.argmax(axis=1).astype(np.int64)
        return y

    for cand in ("target", "label"):
        if cand in df.columns:
            y = df[cand].to_numpy()
            try:
                y = y.astype(np.int64)
            except Exception:
                pass
            return y

    raise ValueError(
        f"Cannot infer labels. Need one-hot columns {DEFAULT_CLASS_COLS} or a 'target'/'label' column."
    )


def stratified_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42, shuffle: bool = True):
    """
    Return list of (train_idx, val_idx) for stratified K-fold without sklearn.
    """
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    per_class_chunks = {}

    for c in classes:
        idx = np.where(y == c)[0]
        if shuffle:
            rng.shuffle(idx)
        per_class_chunks[c] = np.array_split(idx, n_splits)

    all_idx = np.arange(n)
    splits = []

    for fold in range(n_splits):
        val_idx = np.concatenate([per_class_chunks[c][fold] for c in classes]).astype(np.int64)
        if shuffle:
            rng.shuffle(val_idx)

        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = all_idx[mask].astype(np.int64)

        splits.append((train_idx, val_idx))

    return splits


def main():
    root = find_project_root(Path(__file__).resolve().parent)

    labels_path = root / "data" / "dataDir" / "labels" / "official" / "labels.csv"
    if not labels_path.exists():
        # fallback: find any csv under dataDir/labels
        cand_dir = root / "data" / "dataDir" / "labels"
        if cand_dir.exists():
            cands = list(cand_dir.rglob("*.csv"))
            if len(cands) == 1:
                labels_path = cands[0]
            elif len(cands) > 1:
                preferred = [p for p in cands if p.name.lower() == "labels.csv"]
                labels_path = preferred[0] if preferred else cands[0]

    out_dir = root / "data" / "saveDir"
    out_path = out_dir / "indices_isic2019.pkl"

    print(f"[INFO] project root : {root}")
    print(f"[INFO] labels.csv    : {labels_path}")
    print(f"[INFO] output pkl    : {out_path}")

    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found. Expected at: {labels_path}\n"
            "Please place it under: classify/data/dataDir/labels/official/labels.csv"
        )

    df = pd.read_csv(labels_path)

    # IMPORTANT:
    # If your Dataset builds the sample list by sorting image filenames,
    # set this to True so indices match the same order.
    SORT_BY_IMAGE_NAME = False
    if SORT_BY_IMAGE_NAME and "image" in df.columns:
        df = df.sort_values("image").reset_index(drop=True)

    y = infer_labels(df)
    n = len(y)

    print(f"[INFO] rows          : {n}")
    unique, counts = np.unique(y, return_counts=True)
    print("[INFO] class counts  :", dict(zip(unique.tolist(), counts.tolist())))

    n_splits = 5
    splits = stratified_kfold(y, n_splits=n_splits, seed=42, shuffle=True)

    trainIndCV, valIndCV = [], []
    for i, (tr, va) in enumerate(splits):
        trainIndCV.append(tr)
        valIndCV.append(va)
        print(f"[FOLD {i}] train={len(tr)}  val={len(va)}  total={len(tr)+len(va)}")
        assert len(np.intersect1d(tr, va)) == 0
        assert len(np.union1d(tr, va)) == n

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"trainIndCV": trainIndCV, "valIndCV": valIndCV}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[DONE] indices_isic2019.pkl generated successfully.")


if __name__ == "__main__":
    main()
