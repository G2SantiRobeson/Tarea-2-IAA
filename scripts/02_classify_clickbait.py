from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clickbait_rules import DEFAULT_CLICKBAIT_THRESHOLD, RULE_VERSION, classify_headline
from src.eda import write_eda_outputs
from src.io_utils import read_csv, write_csv
from src.text_utils import clean_text, headline_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasifica cada titular como informativo o clickbait usando etiquetado debil explicable."
    )
    parser.add_argument("--input", default="data/raw/headlines_raw.csv", help="CSV creado por 01_web_scraping.py.")
    parser.add_argument("--output", default="data/processed/dataset_clickbait.csv", help="CSV curado y clasificado.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CLICKBAIT_THRESHOLD,
        help="Umbral del score para etiqueta clickbait.",
    )
    parser.add_argument("--review-margin", type=float, default=0.08, help="Margen alrededor del umbral para revision manual.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para split train/validation/test.")
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Cantidad exacta global de filas del dataset final. Si se usa --target-per-source-type, se ignora.",
    )
    parser.add_argument(
        "--target-per-source-type",
        type=int,
        default=1500,
        help="Cantidad exacta de filas por source_type nacional/internacional.",
    )
    parser.add_argument(
        "--target-clickbait",
        type=int,
        default=250,
        help="Cantidad exacta total de casos clickbait en el dataset final.",
    )
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


def select_target_distribution(
    df: pd.DataFrame,
    *,
    target_total: int | None,
    target_per_source_type: int | None,
    target_clickbait: int,
    seed: int,
) -> pd.DataFrame:
    if target_per_source_type is not None:
        if target_per_source_type <= 0:
            raise ValueError("--target-per-source-type debe ser mayor que 0.")
        target_groups = ["nacional", "internacional"]
        target_total = target_per_source_type * len(target_groups)
    elif target_total is None:
        raise ValueError("Debes indicar --target-total o --target-per-source-type.")
    elif target_total <= 0:
        raise ValueError("--target-total debe ser mayor que 0.")

    if target_clickbait < 0 or target_clickbait > target_total:
        raise ValueError("--target-clickbait debe estar entre 0 y el total objetivo.")

    clickbait = df[df["label"] == "clickbait"]
    informative = df[df["label"] == "informativo"]

    if len(clickbait) < target_clickbait:
        raise ValueError(
            "No hay suficientes filas para cumplir la distribucion solicitada: "
            f"clickbait disponibles={len(clickbait)} requeridos={target_clickbait}. "
            "Ajusta --threshold o vuelve a ejecutar el scraping para obtener mas datos."
        )

    clickbait_sample = clickbait.sample(n=target_clickbait, random_state=seed)
    samples = [clickbait_sample]

    if target_per_source_type is not None:
        for offset, group_name in enumerate(target_groups, start=1):
            selected_group_n = int((clickbait_sample["source_type"] == group_name).sum())
            required_informative = target_per_source_type - selected_group_n
            group_informative = informative[informative["source_type"] == group_name]
            if required_informative < 0 or len(group_informative) < required_informative:
                raise ValueError(
                    "No hay suficientes filas informativas para cumplir el balance por grupo: "
                    f"{group_name} disponibles={len(group_informative)} requeridas={required_informative}. "
                    "Vuelve a ejecutar el scraping para obtener mas datos."
                )
            samples.append(group_informative.sample(n=required_informative, random_state=seed + offset))
    else:
        target_informative = target_total - target_clickbait
        if len(informative) < target_informative:
            raise ValueError(
                "No hay suficientes filas informativas para cumplir la distribucion solicitada: "
                f"informativo disponibles={len(informative)} requeridos={target_informative}. "
                "Vuelve a ejecutar el scraping para obtener mas datos."
            )
        samples.append(informative.sample(n=target_informative, random_state=seed + 1))

    selected = pd.concat(samples, ignore_index=True)
    return selected.sample(frac=1, random_state=seed + 2).reset_index(drop=True)


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
    df = select_target_distribution(
        df,
        target_total=args.target_total,
        target_per_source_type=args.target_per_source_type,
        target_clickbait=args.target_clickbait,
        seed=args.seed,
    )
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
