from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clickbait_rules import RULE_VERSION, classify_headline
from src.eda import write_eda_outputs
from src.io_utils import read_csv, write_csv
from src.text_utils import clean_text, headline_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica cada titular como informativo o clickbait usando etiquetado debil explicable."
    )
    parser.add_argument("--input", default="data/raw/headlines_raw.csv", help="CSV creado por 01_web_scraping.py.")
    parser.add_argument("--output", default="data/processed/dataset_clickbait.csv", help="CSV curado y clasificado.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Umbral del score para etiqueta clickbait.")
    parser.add_argument("--review-margin", type=float, default=0.08, help="Margen alrededor del umbral para revision manual.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para split train/validation/test.")
    return parser.parse_args()


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype="object")
    group_cols = [col for col in ["source_type", "label"] if col in df.columns]

    for _, group in df.groupby(group_cols, dropna=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        test_n = max(1, round(n * 0.10)) if n >= 10 else 0
        val_n = max(1, round(n * 0.10)) if n >= 10 else 0
        split.loc[idx[:test_n]] = "test"
        split.loc[idx[test_n : test_n + val_n]] = "validation"
        split.loc[idx[test_n + val_n :]] = "train"

    return split.fillna("train")


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input
    df = read_csv(input_path)

    df["headline"] = df["headline"].map(clean_text)
    df = df[df["headline"] != ""].copy()
    df["headline_key"] = df["headline"].map(headline_key)
    df = df.drop_duplicates("headline_key").copy()

    predictions = df["headline"].map(
        lambda text: classify_headline(text, threshold=args.threshold, review_margin=args.review_margin)
    )
    df["label"] = predictions.map(lambda pred: pred.label)
    df["clickbait_score"] = predictions.map(lambda pred: pred.score)
    df["clickbait_reasons"] = predictions.map(lambda pred: ";".join(pred.reasons))
    df["needs_review"] = predictions.map(lambda pred: pred.needs_review)
    df["labeling_method"] = RULE_VERSION
    df["split"] = assign_splits(df, args.seed)

    ordered_cols = [
        "id",
        "headline",
        "label",
        "source_type",
        "source",
        "author",
        "section",
        "published_at",
        "url",
        "clickbait_score",
        "clickbait_reasons",
        "needs_review",
        "split",
        "labeling_method",
        "collection_method",
        "scraped_at",
    ]
    existing_cols = [col for col in ordered_cols if col in df.columns]
    extra_cols = [col for col in df.columns if col not in existing_cols and col != "headline_key"]
    df = df[existing_cols + extra_cols]

    output_path = write_csv(df, ROOT / args.output)
    review_path = write_csv(
        df.sort_values(["needs_review", "clickbait_score"], ascending=[False, False]).head(250),
        ROOT / "data/processed/manual_review_candidates.csv",
    )
    eda_paths = write_eda_outputs(df, ROOT / "reports")

    print(f"Dataset clasificado guardado en: {output_path}")
    print(f"Muestra para revision manual: {review_path}")
    print("Reportes EDA:")
    for name, path in eda_paths.items():
        print(f"- {name}: {path}")
    print("\nConteo por origen y etiqueta:")
    print(df.groupby(["source_type", "label"]).size().to_string())


if __name__ == "__main__":
    main()
