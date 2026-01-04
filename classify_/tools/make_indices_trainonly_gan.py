# tools/make_indices_trainonly_gan.py
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

def to_np(x):
    return np.array(x, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx_in", required=True)
    ap.add_argument("--labels_official", required=True)
    ap.add_argument("--labels_with_gan", required=True)
    ap.add_argument("--idx_out", required=True)
    args = ap.parse_args()

    N0 = len(pd.read_csv(args.labels_official, encoding="utf-8-sig"))
    N1 = len(pd.read_csv(args.labels_with_gan, encoding="utf-8-sig"))
    if N1 <= N0:
        raise ValueError(f"labels_with_gan must be larger than official. N0={N0}, N1={N1}")

    gan_idx = np.arange(N0, N1, dtype=np.int64)

    with open(args.idx_in, "rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict, got {type(obj)}")

    # ✅keys
    if "trainIndCV" not in obj or "valIndCV" not in obj:
        raise KeyError(f"Expected keys ['trainIndCV','valIndCV'], got {list(obj.keys())}")

    train_cv = obj["trainIndCV"]
    val_cv = obj["valIndCV"]

    if not isinstance(train_cv, list) or not isinstance(val_cv, list):
        raise TypeError(f"trainIndCV/valIndCV should be lists. Got {type(train_cv)}, {type(val_cv)}")
    if len(train_cv) != len(val_cv):
        raise ValueError(f"Fold count mismatch: len(trainIndCV)={len(train_cv)} vs len(valIndCV)={len(val_cv)}")

    new_train = []
    for i, tr in enumerate(train_cv):
        tr_np = to_np(tr)
        new_train.append(np.concatenate([tr_np, gan_idx]))

    new_obj = dict(obj)
    new_obj["trainIndCV"] = new_train
    new_obj["valIndCV"] = val_cv

    Path(args.idx_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.idx_out, "wb") as f:
        pickle.dump(new_obj, f)

    print("✅ saved:", args.idx_out)
    print("Official N0 =", N0, "Total N1 =", N1, "GAN =", N1 - N0)
    print("GAN idx range:", int(gan_idx[0]), "->", int(gan_idx[-1]))
    print("Folds:", len(new_train))

if __name__ == "__main__":
    main()
