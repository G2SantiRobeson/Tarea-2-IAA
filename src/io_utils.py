from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

