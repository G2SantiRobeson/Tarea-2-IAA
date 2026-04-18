from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import read_csv, write_csv
from src.scraping import scrape_fake_news_claims
from src.sources import FAKE_NEWS_SOURCES
from src.text_utils import clean_text, headline_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bonus: obtiene claims falsos desde sitios de fact-checking y arma dataset multiclase."
    )
    parser.add_argument("--target-fake", type=int, default=1000, help="Cantidad objetivo de titulares/claims fake news.")
    parser.add_argument("--max-per-source", type=int, default=300, help="Tope por sitio de fact-checking.")
    parser.add_argument("--delay", type=float, default=0.8, help="Pausa entre descargas de paginas.")
    parser.add_argument("--strict-rating", action="store_true", help="Conserva solo paginas con rating falso/enganoso explicito.")
    parser.add_argument("--output", default="data/raw/fake_news_headlines.csv", help="CSV de fake news extraidas.")
    parser.add_argument(
        "--base-dataset",
        default="data/processed/dataset_clickbait.csv",
        help="Dataset binario para combinar en version multiclase.",
    )
    parser.add_argument(
        "--combined-output",
        default="data/processed/dataset_multiclase_bonus.csv",
        help="CSV final con informativo/clickbait/fake_news.",
    )
    return parser.parse_args()


def normalize_fake_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["headline"] = df["headline"].map(clean_text)
    df = df[df["headline"] != ""].copy()
    df["headline_key"] = df["headline"].map(headline_key)
    df = df.drop_duplicates("headline_key").copy()
    df["label"] = "fake_news"
    df["source_type"] = "fake_news"
    df["clickbait_score"] = ""
    df["clickbait_reasons"] = ""
    df["needs_review"] = False
    df["labeling_method"] = "claimreview-factcheck-v1.0"
    df["split"] = "train"
    return df.drop(columns=["headline_key"])


def main() -> None:
    args = parse_args()
    records = scrape_fake_news_claims(
        FAKE_NEWS_SOURCES,
        target_total=args.target_fake,
        max_per_source=args.max_per_source,
        delay_seconds=args.delay,
        keep_unrated=not args.strict_rating,
    )
    fake_df = normalize_fake_frame(pd.DataFrame(records))
    fake_output = write_csv(fake_df, ROOT / args.output)
    print(f"Fake news guardadas en: {fake_output}")
    if fake_df.empty:
        print("No se obtuvieron fake news. Prueba sin --strict-rating o aumenta max-per-source.")
        return

    base_path = ROOT / args.base_dataset
    if base_path.exists():
        base_df = read_csv(base_path)
        common_cols = sorted(set(base_df.columns).union(fake_df.columns))
        combined = pd.concat(
            [base_df.reindex(columns=common_cols), fake_df.reindex(columns=common_cols)],
            ignore_index=True,
        )
        combined = combined.drop_duplicates(subset=["headline"], keep="first")
        combined_output = write_csv(combined, ROOT / args.combined_output)
        print(f"Dataset multiclase guardado en: {combined_output}")
        print("\nConteo por etiqueta:")
        print(combined.groupby("label").size().to_string())
    else:
        print(f"No existe {base_path}; se genero solo el CSV del bonus.")


if __name__ == "__main__":
    main()

