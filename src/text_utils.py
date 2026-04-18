from __future__ import annotations

import hashlib
import html
import re
import unicodedata


def clean_text(value: object) -> str:
    """Normalize whitespace and HTML entities without changing semantic content."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    if "Ã" in text or "Â" in text:
        try:
            text = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            pass
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n\"'")


def strip_common_title_suffix(title: str) -> str:
    """Remove common site-name suffixes from page titles."""
    title = clean_text(title)
    for sep in (" | ", " - ", " – ", " — "):
        if sep in title:
            left, right = title.rsplit(sep, 1)
            if len(left) >= 15 and len(right.split()) <= 6:
                return clean_text(left)
    return title


def normalize_for_matching(value: object) -> str:
    """Lowercase, remove accents and keep a compact representation for matching."""
    text = clean_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9ñ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def headline_key(headline: object) -> str:
    """Stable key for headline deduplication."""
    normalized = normalize_for_matching(headline)
    normalized = re.sub(r"\b(video|fotos?|minuto a minuto|en vivo)\b", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "||".join(clean_text(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def looks_like_valid_headline(headline: object) -> bool:
    text = clean_text(headline)
    if len(text) < 18:
        return False
    if len(text.split()) < 4:
        return False
    lower = normalize_for_matching(text)
    noisy = (
        "cookie",
        "suscribete",
        "newsletter",
        "iniciar sesion",
        "publicidad",
        "menu",
        "buscar",
    )
    return not any(token in lower for token in noisy)
