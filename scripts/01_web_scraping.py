from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import write_csv
from src.scraping import scrape_sources
from src.sources import NEWS_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Obtiene titulares nacionales e internacionales mediante RSS, sitemap y paginas de articulo."
    )
    parser.add_argument("--target-per-group", type=int, default=1000, help="Minimo esperado por grupo: nacional/internacional.")
    parser.add_argument("--max-per-source", type=int, default=350, help="Tope de titulares por portal.")
    parser.add_argument("--delay", type=float, default=0.8, help="Pausa entre descargas de articulos.")
    parser.add_argument("--rss-only", action="store_true", help="Usa solo feeds RSS/Atom y no baja articulos desde sitemaps.")
    parser.add_argument(
        "--no-article-pages",
        action="store_true",
        help="No descarga paginas completas; usa solo RSS y titulos presentes en sitemaps.",
    )
    parser.add_argument("--output", default="data/raw/headlines_raw.csv", help="Ruta CSV de salida.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = scrape_sources(
        NEWS_SOURCES,
        target_per_group=args.target_per_group,
        max_per_source=args.max_per_source,
        delay_seconds=args.delay,
        include_sitemaps=not args.rss_only,
        fetch_article_pages=not args.no_article_pages,
    )
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["source_type", "source", "published_at", "headline"], na_position="last")
    output_path = write_csv(df, ROOT / args.output)

    print(f"CSV guardado en: {output_path}")
    if df.empty:
        print("No se obtuvieron registros. Revisa conexion, dependencias y disponibilidad de las fuentes.")
        return
    print("Conteo por grupo:")
    print(df.groupby("source_type").size().to_string())
    print("\nConteo por portal:")
    print(df.groupby(["source_type", "source"]).size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
