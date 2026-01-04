# tools/rename_gan_inplace.py
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image, ImageOps

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

def norm_class(x: str) -> str:
    x = str(x).strip().upper()
    x = x.replace("AKIEC", "AK")
    x = x.replace("VASCULAR", "VASC")
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/dataDir/images/gan", help="GAN image folder to overwrite")
    ap.add_argument("--labels", default="data/dataDir/labels/GAN_labels.csv", help="CSV with filename,class,age,sex,site")
    ap.add_argument("--backup", action="store_true", help="backup originals before overwrite (recommended)")
    ap.add_argument("--no_backup", action="store_true", help="DO NOT backup originals (dangerous)")
    ap.add_argument("--skip_missing", action="store_true", help="skip rows whose files are missing (not recommended)")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    labels_path = Path(args.labels)

    if not img_dir.exists():
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels not found: {labels_path}")

    # Decide backup behavior
    do_backup = args.backup and (not args.no_backup)
    if args.no_backup:
        do_backup = False

    # Read labels
    df = pd.read_csv(labels_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    required = {"filename", "class"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"labels must contain columns {required}, got {list(df.columns)}")

    df["filename"] = df["filename"].astype(str).str.strip()
    df["class"] = df["class"].apply(norm_class)

    bad = df.loc[~df["class"].isin(CLASSES), "class"].unique().tolist()
    if bad:
        raise ValueError(f"Unknown class values: {bad}. Must be subset of {CLASSES}")

    # Deterministic order: sort by class then filename
    df = df.sort_values(["class", "filename"], kind="mergesort").reset_index(drop=True)

    # Prepare temp output
    tmp_dir = img_dir.parent / (img_dir.name + "_tmp_convert")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Convert + rename into tmp_dir
    counters = {c: 0 for c in CLASSES}
    rows_map = []
    missing_files = []

    for _, row in df.iterrows():
        old_name = row["filename"]
        cls = row["class"]
        src = img_dir / old_name

        if not src.exists():
            missing_files.append(old_name)
            if args.skip_missing:
                continue
            else:
                continue  # collect all missing, then error after loop

        counters[cls] += 1
        new_id = f"GAN_{cls}_{counters[cls]:06d}"
        dst = tmp_dir / f"{new_id}.jpg"

        im = Image.open(src)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.save(dst, format="JPEG", quality=95, optimize=True)

        rows_map.append({
            "filename": old_name,    
            "class": cls,
            "new_id": new_id,
            "new_filename": dst.name,
        })

    if missing_files and not args.skip_missing:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sample = missing_files[:20]
        raise FileNotFoundError(
            f"Found {len(missing_files)} filenames listed in CSV but missing in {img_dir}.\n"
            f"Examples: {sample}\n"
            f"Fix the files or rerun with --skip_missing (not recommended)."
        )

    map_df = pd.DataFrame(rows_map)
    if map_df.empty:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("No images converted. Check your CSV/paths.")

    # Save mapping + renamed GT/meta
    labels_dir = labels_path.parent

    map_csv = labels_dir / "gan_rename_map.csv"
    map_df.to_csv(map_csv, index=False, encoding="utf-8-sig")

    gt = pd.DataFrame({"image": map_df["new_id"]})
    for c in CLASSES:
        gt[c] = (map_df["class"] == c).astype(int)
    gt_path = labels_dir / "gan_groundtruth_onehot_renamed.csv"
    gt.to_csv(gt_path, index=False, encoding="utf-8-sig")

    joined = df.merge(map_df[["filename", "new_id"]], on="filename", how="inner")

    meta = pd.DataFrame({"image": joined["new_id"]})

    meta["age_approx"] = pd.to_numeric(joined["age"], errors="coerce") if "age" in joined.columns else pd.NA

    if "sex" in joined.columns:
        sex = joined["sex"].astype(str).str.strip().str.lower()
    else:
        sex = pd.Series(["unknown"] * len(joined))
    sex = sex.where(sex.isin(["male", "female"]), "unknown")
    meta["sex"] = sex

    if "site" in joined.columns:
        site = joined["site"].astype(str).str.strip().str.lower()
    else:
        site = pd.Series(["unknown"] * len(joined))

    site_map = {"anterior": "anterior torso", "lateral torso": "torso"}
    site = site.replace(site_map)

    allowed_sites = {
        "unknown", "head/neck", "upper extremity", "lower extremity",
        "torso", "anterior torso", "posterior torso", "palms/soles", "oral/genital"
    }
    site = site.where(site.isin(allowed_sites), "unknown")
    meta["anatom_site_general_challenge"] = site

    meta_path = labels_dir / "gan_metadata_renamed.csv"
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    # Overwrite gan folder
    if do_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = img_dir.parent / f"{img_dir.name}_backup_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in img_dir.iterdir():
            if p.is_file():
                shutil.move(str(p), str(backup_dir / p.name))
        print(f"✅ Originals backed up to: {backup_dir}")
    else:
        for p in img_dir.iterdir():
            if p.is_file():
                p.unlink()

    for p in tmp_dir.iterdir():
        shutil.move(str(p), str(img_dir / p.name))

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n✅ DONE (in-place overwrite).")
    print("Final images folder:", img_dir.resolve())
    print("Example output files:", sorted([p.name for p in img_dir.iterdir() if p.is_file()])[:5])
    print("\nSaved:")
    print(" -", map_csv.resolve())
    print(" -", gt_path.resolve())
    print(" -", meta_path.resolve())
    print("\nPer-class converted counts:", counters)
    print("Total converted:", len(map_df))

if __name__ == "__main__":
    main()
