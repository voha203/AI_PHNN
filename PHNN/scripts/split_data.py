import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import os


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_files(df, src_dir: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying to {dst_dir.name}"):
        src = src_dir / row["filename"]
        if not src.exists():
            print(f"Warning: source file missing: {src}")
            continue
        dst = dst_dir / row["filename"]
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=Path, default=Path("dataset/labels.csv"), help="CSV produced by create_lable.py")
    p.add_argument("--images", type=Path, default=Path("dataset/raw"), help="Folder with images referenced in labels CSV")
    p.add_argument("--out", type=Path, default=Path("dataset/processed"), help="Output processed folder (train/val/test)")
    p.add_argument("--train", type=float, default=0.7, help="Proportion for training set")
    p.add_argument("--val", type=float, default=0.15, help="Proportion for validation set")
    p.add_argument("--test", type=float, default=0.15, help="Proportion for test set")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--min-text-length", type=int, default=1, help="Minimum OCR text length to include in stratify grouping (0 to include all)")
    args = p.parse_args()

    if not args.labels.exists():
        raise SystemExit(f"Labels CSV not found: {args.labels}")

    if not args.images.exists():
        raise SystemExit(f"Images folder not found: {args.images}")

    df = pd.read_csv(args.labels)
    if "filename" not in df.columns:
        raise SystemExit("labels.csv must have a 'filename' column")

    if "language" in df.columns and df["language"].notna().any():
        strat_col = "language"
        df[strat_col] = df[strat_col].fillna("unknown").astype(str)
    else:
        strat_col = None

    df["src_exists"] = df["filename"].map(lambda fn: (args.images / fn).exists())
    missing = df[~df["src_exists"]]
    if not missing.empty:
        print(f"Warning: {len(missing)} entries in labels.csv have missing files; they will be skipped.")
        df = df[df["src_exists"]].copy()

    train_p = args.train
    val_p = args.val
    test_p = args.test
    total = train_p + val_p + test_p
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("train+val+test proportions must sum to 1.0")

    if strat_col:
        train_df, temp_df = train_test_split(df, train_size=train_p, stratify=df[strat_col], random_state=args.seed)
        temp_ratio = val_p / (val_p + test_p)
        val_df, test_df = train_test_split(temp_df, train_size=temp_ratio, stratify=temp_df[strat_col], random_state=args.seed)
    else:
        train_df, temp_df = train_test_split(df, train_size=train_p, random_state=args.seed)
        temp_ratio = val_p / (val_p + test_p)
        val_df, test_df = train_test_split(temp_df, train_size=temp_ratio, random_state=args.seed)

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    out = args.out
    train_out = out / "train"
    val_out = out / "val"
    test_out = out / "test"
    copy_files(train_df, args.images, train_out)
    copy_files(val_df, args.images, val_out)
    copy_files(test_df, args.images, test_out)
    ensure_dir(out)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    print(f"Saved split CSVs to {out}. Images copied to {train_out}, {val_out}, {test_out}")


if __name__ == "__main__":
    main()