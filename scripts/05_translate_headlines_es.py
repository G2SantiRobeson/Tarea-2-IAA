from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, LangDetectException, detect_langs
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import write_csv
from src.text_utils import clean_text

DetectorFactory.seed = 42

DEFAULT_PATHS = [
    "data/processed/dataset_clickbait.csv",
    "data/processed/manual_review_candidates.csv",
    "data/raw/fake_news_headlines.csv",
    "data/processed/dataset_multiclase_bonus.csv",
]

FORCE_TRANSLATE_SOURCES = {
    "FactCheck.org",
    "PolitiFact",
    "Snopes",
    "The Onion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traduce al espanol la columna headline y conserva el titular original."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_PATHS,
        help="CSVs a actualizar. Por defecto traduce los datasets principales.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.75,
        help="Probabilidad minima de deteccion para traducir titulares no espanoles.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Pausa entre llamadas al traductor.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra resumen sin escribir archivos.",
    )
    return parser.parse_args()


def detect_language(text: str) -> tuple[str, float]:
    text = clean_text(text)
    if len(text) < 20:
        return "short", 0.0
    try:
        detected = detect_langs(text)
    except LangDetectException:
        return "unknown", 0.0
    if not detected:
        return "unknown", 0.0
    top = detected[0]
    return top.lang, float(top.prob)


def should_translate(row: pd.Series, *, min_prob: float) -> tuple[bool, str]:
    source = clean_text(row.get("source"))
    detected = clean_text(row.get("headline_language_detected"))
    prob = float(row.get("headline_language_probability") or 0.0)

    if source in FORCE_TRANSLATE_SOURCES:
        return True, "forced_source"
    if detected not in {"", "es", "short", "unknown"} and prob >= min_prob:
        return True, "detected_non_es"
    return False, "kept"


def translate_text(
    text: str,
    translator: GoogleTranslator,
    cache: dict[str, str],
    *,
    delay_seconds: float,
) -> tuple[str, bool]:
    text = clean_text(text)
    if not text:
        return text, False
    if text in cache:
        return cache[text], cache[text] != text
    try:
        translated = clean_text(translator.translate(text))
    except Exception as exc:  # noqa: BLE001 - external translator can fail for many transient reasons.
        print(f"[WARN] No se pudo traducir: {text[:90]} ({exc})")
        cache[text] = text
        return text, False
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    cache[text] = translated or text
    return cache[text], translated != text


def translate_frame(
    df: pd.DataFrame,
    translator: GoogleTranslator,
    cache: dict[str, str],
    *,
    min_prob: float,
    delay_seconds: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if "headline" not in df.columns:
        raise ValueError("El CSV no contiene columna headline.")

    df = df.copy()
    if "headline_original" not in df.columns:
        df.insert(df.columns.get_loc("headline") + 1, "headline_original", df["headline"])
    else:
        df["headline_original"] = df["headline_original"].where(
            df["headline_original"].astype(str).str.strip() != "",
            df["headline"],
        )

    detections = df["headline_original"].map(detect_language)
    df["headline_language_detected"] = detections.map(lambda item: item[0])
    df["headline_language_probability"] = detections.map(lambda item: round(item[1], 6))

    decisions = df.apply(lambda row: should_translate(row, min_prob=min_prob), axis=1)
    df["_translate"] = decisions.map(lambda item: item[0])
    df["headline_translation_reason"] = decisions.map(lambda item: item[1])
    df["headline_translation_method"] = ""

    stats = {
        "rows": len(df),
        "to_translate": int(df["_translate"].sum()),
        "translated": 0,
        "failed_or_unchanged": 0,
        "kept": int((~df["_translate"]).sum()),
    }

    for idx in tqdm(df.index[df["_translate"]], desc="Traduciendo titulares"):
        original = df.at[idx, "headline_original"]
        translated, changed = translate_text(
            original,
            translator,
            cache,
            delay_seconds=delay_seconds,
        )
        df.at[idx, "headline"] = translated
        df.at[idx, "headline_translation_method"] = "GoogleTranslator(auto->es)"
        if changed:
            stats["translated"] += 1
        else:
            stats["failed_or_unchanged"] += 1

    kept_mask = ~df["_translate"]
    df.loc[kept_mask, "headline"] = df.loc[kept_mask, "headline_original"]
    df.loc[kept_mask, "headline_translation_method"] = "kept_original"
    df = df.drop(columns=["_translate"])
    return df, stats


def main() -> None:
    args = parse_args()
    translator = GoogleTranslator(source="auto", target="es")
    cache: dict[str, str] = {}

    for raw_path in args.paths:
        path = ROOT / raw_path
        if not path.exists():
            print(f"[WARN] No existe {path}; se omite.")
            continue
        df = pd.read_csv(path)
        translated, stats = translate_frame(
            df,
            translator,
            cache,
            min_prob=args.min_prob,
            delay_seconds=args.delay,
        )
        print(f"\n{path}")
        print(stats)
        if not args.dry_run:
            write_csv(translated, path)
            print(f"Actualizado: {path}")


if __name__ == "__main__":
    main()
