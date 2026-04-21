from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import write_csv
from src.text_utils import clean_text

VALID_LABELS = {"informativo", "clickbait", "fake_news"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica etiquetas revisadas manualmente al dataset base y derivados."
    )
    parser.add_argument(
        "--reviewed",
        default="data/hand_processed/manual_review_candidates_REVIEWED.csv",
        help="CSV revisado manualmente con columnas id y label.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[
            "data/processed/dataset_clickbait.csv",
            "data/processed/manual_review_candidates.csv",
            "data/processed/dataset_multiclase_bonus.csv",
        ],
        help="CSVs donde aplicar las etiquetas corregidas.",
    )
    parser.add_argument(
        "--method",
        default="manual-review-v1.0",
        help="Valor para labeling_method en filas corregidas/revisadas.",
    )
    return parser.parse_args()


def load_reviewed(path: Path) -> pd.DataFrame:
    reviewed = pd.read_csv(path)
    missing = {"id", "label"} - set(reviewed.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")

    reviewed = reviewed.copy()
    reviewed["id"] = reviewed["id"].map(clean_text)
    reviewed["label"] = reviewed["label"].map(lambda value: clean_text(value).lower())
    reviewed = reviewed[reviewed["id"] != ""].drop_duplicates("id", keep="last")

    invalid = sorted(set(reviewed["label"]) - VALID_LABELS)
    if invalid:
        raise ValueError(f"Labels invalidos en revision manual: {invalid}")
    return reviewed[["id", "label"]]


def apply_review(target_path: Path, reviewed: pd.DataFrame, method: str) -> dict[str, int]:
    df = pd.read_csv(target_path)
    if "id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{target_path} debe contener columnas id y label.")

    review_map = dict(zip(reviewed["id"], reviewed["label"], strict=True))
    target_ids = df["id"].map(clean_text)
    mask = target_ids.isin(review_map)
    before = df.loc[mask, "label"].map(lambda value: clean_text(value).lower())
    after = target_ids[mask].map(review_map)
    changed = before.ne(after)

    df.loc[mask, "label"] = after.values
    if "needs_review" in df.columns:
        df.loc[mask, "needs_review"] = False
    if "labeling_method" in df.columns:
        df.loc[mask, "labeling_method"] = method

    write_csv(df, target_path)
    return {
        "rows": len(df),
        "matched_reviewed_rows": int(mask.sum()),
        "labels_changed": int(changed.sum()),
    }


def main() -> None:
    args = parse_args()
    reviewed_path = ROOT / args.reviewed
    reviewed = load_reviewed(reviewed_path)
    print(f"Revision manual: {reviewed_path}")
    print(reviewed["label"].value_counts().to_string())

    for raw_target in args.targets:
        target_path = ROOT / raw_target
        if not target_path.exists():
            print(f"[WARN] No existe {target_path}; se omite.")
            continue
        stats = apply_review(target_path, reviewed, args.method)
        print(f"\nActualizado: {target_path}")
        print(stats)


if __name__ == "__main__":
    main()
