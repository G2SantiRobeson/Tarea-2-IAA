from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eda import write_eda_outputs
from src.io_utils import read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenera reportes EDA de portales/autores desde el dataset clasificado.")
    parser.add_argument("--input", default="data/processed/dataset_clickbait.csv")
    parser.add_argument("--reports-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_csv(ROOT / args.input)
    paths = write_eda_outputs(df, ROOT / args.reports_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

