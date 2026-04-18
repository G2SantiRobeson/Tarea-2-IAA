from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import write_csv


def dataframe_to_markdown(df: pd.DataFrame, *, max_rows: int = 12) -> str:
    if df.empty:
        return "_Sin registros suficientes._"
    preview = df.head(max_rows).copy()
    preview = preview.fillna("")
    columns = list(preview.columns)
    rows = [[str(value) for value in row] for row in preview.to_numpy()]
    widths = [
        max(len(str(col)), *(len(row[i]) for row in rows)) if rows else len(str(col))
        for i, col in enumerate(columns)
    ]
    header = "| " + " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def clickbait_rate_table(df: pd.DataFrame, group_cols: list[str], *, min_items: int = 5) -> pd.DataFrame:
    table = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("headline", "size"),
            clickbait_n=("label", lambda s: int((s == "clickbait").sum())),
            informative_n=("label", lambda s: int((s == "informativo").sum())),
            avg_clickbait_score=("clickbait_score", "mean"),
        )
        .reset_index()
    )
    table = table[table["n"] >= min_items].copy()
    table["clickbait_rate"] = table["clickbait_n"] / table["n"]
    return table.sort_values(["clickbait_rate", "n"], ascending=[False, False])


def write_markdown_summary(
    df: pd.DataFrame,
    source_table: pd.DataFrame,
    author_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts = df.groupby(["source_type", "label"]).size().unstack(fill_value=0)
    lines = [
        "# Resumen EDA Entrega 1",
        "",
        "## Cobertura del dataset",
        "",
        dataframe_to_markdown(label_counts.reset_index(), max_rows=20),
        "",
        "## Portales con mayor tasa de clickbait",
        "",
        dataframe_to_markdown(source_table, max_rows=12),
        "",
        "## Autores con mayor tasa de clickbait",
        "",
    ]
    if author_table.empty:
        lines.append("No hay autores suficientes para calcular una tasa estable con el umbral usado.")
    else:
        lines.append(dataframe_to_markdown(author_table, max_rows=12))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_eda_outputs(df: pd.DataFrame, reports_dir: str | Path = "reports") -> dict[str, Path]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    source_table = clickbait_rate_table(df, ["source_type", "source"], min_items=10)
    authors = df.copy()
    authors["author"] = authors["author"].fillna("").astype(str).str.strip()
    authors = authors[authors["author"] != ""]
    author_table = clickbait_rate_table(authors, ["source_type", "source", "author"], min_items=5)

    paths = {
        "source_rates": write_csv(source_table, reports_dir / "clickbait_by_source.csv"),
        "author_rates": write_csv(author_table, reports_dir / "clickbait_by_author.csv"),
        "summary": write_markdown_summary(
            df,
            source_table,
            author_table,
            reports_dir / "eda_entrega1_summary.md",
        ),
    }
    return paths
