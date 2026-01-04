#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ISIC2019 meta-data pickle expected by *_meta configs (ss_meta/rr_meta).

Output keys:
  im_name (N,), age_num (N,), age_oh (N, 1+nbins),
  sex_oh (N, 1+Ksex), loc_oh (N, 1+Kloc)

Notes:
- One-hot vectors include an explicit NaN bucket at index 0.
- If your cfg sets encode_nan=False, the loader will drop that first bucket (index 0).
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd


def _normalize_token(v) -> str:
    """Normalize categorical token: strip/lower, treat common unknown tokens as ''."""
    if pd.isna(v):
        return ""
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "null", "unknown", "na", "n/a"}:
        return ""
    return s


def one_hot_with_nan(values: pd.Series, categories: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    categories excludes NaN; we reserve index 0 for NaN/unknown.
    Unknown/unseen values -> NaN bucket.
    """
    cat_to_idx = {c: i + 1 for i, c in enumerate(categories)}  # start at 1
    out = np.zeros((len(values), 1 + len(categories)), dtype=np.float32)

    for i, v in enumerate(values):
        key = _normalize_token(v)
        if key == "":
            out[i, 0] = 1.0
        elif key in cat_to_idx:
            out[i, cat_to_idx[key]] = 1.0
        else:
            out[i, 0] = 1.0
    return out, cat_to_idx


def age_one_hot(values: pd.Series, bins: List[float]) -> np.ndarray:
    """
    bins: list of edges, e.g., [0,10,20,...,1000]
    Output: shape (N, 1+(len(bins)-1)), with NaN bucket at 0.
    Robust to non-numeric strings (treated as NaN).
    """
    out = np.zeros((len(values), 1 + (len(bins) - 1)), dtype=np.float32)

    # Ensure numeric; non-numeric -> NaN
    v_num = pd.to_numeric(values, errors="coerce")

    for i, v in enumerate(v_num):
        if pd.isna(v):
            out[i, 0] = 1.0
        else:
            a = float(v)
            idx = np.digitize(a, bins, right=False)  # 1..len(bins)-1
            idx = max(1, min(idx, len(bins) - 1))
            out[i, idx] = 1.0
    return out


def pick_existing_file(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def main():
    script_dir = Path(__file__).resolve().parent  # you put this script under classify/

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="",
        help="Path to metadata_train.csv (or metadata_train.csv.csv). "
             "If empty, will try common names in the same folder as this script.",
    )
    ap.add_argument(
        "--out_pkl",
        default="",
        help="Output .pkl path. If empty, defaults to "
             "./data/dataDir/meta_data/official/isic2019_meta.pkl (relative to this script).",
    )
    args = ap.parse_args()

    # -------- resolve input csv --------
    if args.in_csv.strip():
        in_csv = Path(args.in_csv).expanduser().resolve()
    else:
        in_csv = pick_existing_file([
            script_dir / "metadata_train.csv",
            script_dir / "metadata_train.csv.csv",
            script_dir / "metadata.csv",
        ])
        if in_csv is None:
            raise FileNotFoundError(
                f"Cannot find metadata csv in {script_dir}. "
                f"Please put metadata_train.csv(.csv) here or pass --in_csv <path>."
            )

    # -------- resolve output pkl --------
    if args.out_pkl.strip():
        out_pkl = Path(args.out_pkl).expanduser().resolve()
    else:
        # IMPORTANT: align with your ss_meta default dataDir = pathBase + '/data/dataDir'
        # and pathBase == classify (script_dir)
        out_pkl = (script_dir / "data" / "dataDir" / "meta_data" / "official" / "isic2019_meta.pkl").resolve()

    df = pd.read_csv(in_csv)

    # Required id column
    if "image" not in df.columns:
        raise ValueError(
            "metadata csv must have an 'image' column. "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure im_name matches labels keys (no extension)
    im_name = (
        df["image"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
        .values
    )

    # -------- age_num: fill NaN with mean --------
    if "age_approx" in df.columns:
        age_raw_num = pd.to_numeric(df["age_approx"], errors="coerce")
        age_mean = float(age_raw_num.mean()) if not np.isnan(age_raw_num.mean()) else 0.0
        age_num = age_raw_num.fillna(age_mean).astype(np.float32).values
    else:
        age_mean = 0.0
        age_num = np.zeros(len(df), dtype=np.float32)

    # -------- age_oh --------
    age_bins = list(range(0, 101, 10)) + [1000]  # 0-10,...,90-100, 100+
    age_source = df["age_approx"] if "age_approx" in df.columns else pd.Series([np.nan] * len(df))
    age_oh = age_one_hot(age_source, age_bins)

    # -------- sex_oh --------
    sex_raw = df["sex"] if "sex" in df.columns else pd.Series([np.nan] * len(df))
    sex_norm = sex_raw.map(_normalize_token)
    sex_categories = sorted({x for x in sex_norm.unique() if x != ""})

    # stable order if present
    stable = [c for c in ["male", "female"] if c in sex_categories]
    sex_categories = stable + [c for c in sex_categories if c not in stable]

    sex_oh, _ = one_hot_with_nan(sex_raw, sex_categories)

    # -------- loc_oh (support multiple column names) --------
    loc_col = None
    for cand in ["anatom_site_general_challenge", "anatom_site_general", "anatom_site"]:
        if cand in df.columns:
            loc_col = cand
            break

    if loc_col is None:
        loc_raw = pd.Series([np.nan] * len(df))
        loc_categories = []
        loc_oh, _ = one_hot_with_nan(loc_raw, loc_categories)
        print(
            "[WARN] No location column found. Expected one of "
            "['anatom_site_general_challenge','anatom_site_general','anatom_site']. "
            "loc_oh will only contain NaN bucket."
        )
    else:
        loc_raw = df[loc_col]
        loc_norm = loc_raw.map(_normalize_token)
        loc_categories = sorted({x for x in loc_norm.unique() if x != ""})
        loc_oh, _ = one_hot_with_nan(loc_raw, loc_categories)

    meta = {
        "im_name": im_name,
        "age_num": age_num,
        "age_oh": age_oh,
        "sex_oh": sex_oh,
        "loc_oh": loc_oh,
        "age_mean_filled": age_mean,
        "sex_categories": sex_categories,
        "loc_categories": loc_categories,
        "source_csv": str(in_csv),
        "loc_col_used": loc_col if loc_col is not None else "",
    }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Wrote:", out_pkl)
    print("Source CSV:", in_csv)
    print("N =", len(im_name))
    print("age_num dim =", meta["age_num"].shape)
    print("age_oh  dim =", meta["age_oh"].shape)
    print("sex_oh dim  =", meta["sex_oh"].shape, "categories =", sex_categories)
    print("loc_oh dim  =", meta["loc_oh"].shape, "K =", len(loc_categories), "loc_col =", meta["loc_col_used"])
    print()
    print("For ss_meta (encode_nan=False), set meta_feature_sizes like:")
    print("  age_num = 1")
    print("  sex_oh  = sex_oh_dim - 1")
    print("  loc_oh  = loc_oh_dim - 1")


if __name__ == "__main__":
    main()
