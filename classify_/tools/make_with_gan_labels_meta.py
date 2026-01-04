# tools/make_with_gan_labels_meta.py
import argparse
from pathlib import Path
import pandas as pd

def rcsv(p):
    return pd.read_csv(p, encoding="utf-8-sig")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official_labels", default="data/dataDir/labels/official/labels.csv")
    ap.add_argument("--official_meta", default="metadata_train.csv.csv")
    ap.add_argument("--gan_labels", default="data/dataDir/labels/gan_groundtruth_onehot_renamed.csv")
    ap.add_argument("--gan_meta", default="data/dataDir/labels/gan_metadata_renamed.csv")
    ap.add_argument("--out_labels", default="data/dataDir/labels/labels_with_gan.csv")
    ap.add_argument("--out_meta", default="data/dataDir/labels/metadata_train_with_gan.csv")
    args = ap.parse_args()

    off_lab = rcsv(args.official_labels)
    gan_lab = rcsv(args.gan_labels)

    # --- labels: align columns (whatever columns official has, GAN should fill; e.g., UNK) ---
    if "image" not in off_lab.columns:
        raise ValueError(f"official labels must have 'image' col, got {off_lab.columns.tolist()}")

    if "image" not in gan_lab.columns:
        # Compatibility: GAN labels may use 'image_id'
        if "image_id" in gan_lab.columns:
            gan_lab = gan_lab.rename(columns={"image_id":"image"})
        else:
            raise ValueError(f"gan labels must have 'image' col, got {gan_lab.columns.tolist()}")

    for c in off_lab.columns:
        if c not in gan_lab.columns:
            if c == "image":
                continue
            gan_lab[c] = 0

    gan_lab = gan_lab[off_lab.columns]
    lab_all = pd.concat([off_lab, gan_lab], ignore_index=True)

    if lab_all["image"].duplicated().any():
        dup = lab_all.loc[lab_all["image"].duplicated(), "image"].head(10).tolist()
        raise ValueError(f"Duplicate image ids after labels merge, examples: {dup}")

    Path(args.out_labels).parent.mkdir(parents=True, exist_ok=True)
    lab_all.to_csv(args.out_labels, index=False, encoding="utf-8-sig")
    print("✅ labels_with_gan:", args.out_labels, "rows=", len(lab_all))

    # --- metadata: align columns and append ---
    off_meta = rcsv(args.official_meta)
    gan_meta = rcsv(args.gan_meta)

    # Find the id column (prefer 'image')
    id_col = "image" if "image" in off_meta.columns else off_meta.columns[0]
    if id_col not in gan_meta.columns and "image" in gan_meta.columns:
        gan_meta = gan_meta.rename(columns={"image": id_col})

    if id_col not in gan_meta.columns:
        raise ValueError(f"gan meta missing id col '{id_col}', gan_meta cols={gan_meta.columns.tolist()}")

    for c in off_meta.columns:
        if c not in gan_meta.columns:
            gan_meta[c] = pd.NA
    gan_meta = gan_meta[off_meta.columns]

    meta_all = pd.concat([off_meta, gan_meta], ignore_index=True)

    if meta_all[id_col].duplicated().any():
        dup = meta_all.loc[meta_all[id_col].duplicated(), id_col].head(10).tolist()
        raise ValueError(f"Duplicate image ids after meta merge, examples: {dup}")

    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    meta_all.to_csv(args.out_meta, index=False, encoding="utf-8-sig")
    print("✅ metadata_train_with_gan:", args.out_meta, "rows=", len(meta_all))

if __name__ == "__main__":
    main()
