from __future__ import annotations

import re
from dataclasses import dataclass

from .text_utils import clean_text, normalize_for_matching


RULE_VERSION = "weak-rules-v1.2"
DEFAULT_CLICKBAIT_THRESHOLD = 0.35


SENSATIONAL_WORDS = {
    "impactante",
    "increible",
    "insolito",
    "sorprendente",
    "viral",
    "escandalo",
    "polemica",
    "brutal",
    "terrible",
    "dramatico",
    "devastador",
    "conmovedor",
    "indignante",
    "secreto",
    "revelador",
    "filtran",
    "filtrado",
    "revelan",
    "urgente",
    "ultima hora",
    "alerta",
    "inedito",
    "misterioso",
    "oculto",
    "prohibido",
    "shock",
    "shocking",
    "unbelievable",
}


CURIOSITY_PHRASES = {
    "no creeras",
    "nadie esperaba",
    "lo que",
    "lo que nadie",
    "lo que se sabe",
    "esto es lo que",
    "que debes saber",
    "esta es la razon",
    "esta fue la razon",
    "asi fue",
    "asi quedo",
    "asi reacciono",
    "que paso",
    "por que",
    "la verdad sobre",
    "el secreto de",
    "mira como",
    "mira lo que",
    "you wont believe",
    "what happened next",
}


CLICKBAIT_FRAMING_PHRASES = {
    "adios a",
    "asi es",
    "asi fue",
    "asi luce",
    "asi quedo",
    "asi reacciono",
    "como ver",
    "conoce los detalles",
    "donde ver",
    "el secreto",
    "el truco",
    "esta es",
    "esta es la razon",
    "estas son",
    "este es",
    "estos son",
    "fecha exacta",
    "hora y donde",
    "la verdad que",
    "la verdad sobre",
    "lo que dice",
    "lo que muchos",
    "los mejores",
    "los peores",
    "nadie lo esperaba",
    "no lo esperaba",
    "no lo esperabamos",
    "no te puedes perder",
    "por esta razon",
    "que debes saber",
    "que dice",
    "quien es",
    "quienes son",
    "revisa aqui",
    "sin que te des cuenta",
    "te contamos",
    "todo lo que debes saber",
}


REACTION_PHRASES = {
    "aclaro supuesto",
    "desato reacciones",
    "duro descargo",
    "estallo tras",
    "fuerte cruce",
    "genero ola de reacciones",
    "lluvia de criticas",
    "no paso desapercibido",
    "rompio el silencio",
    "salio jugando",
    "sin filtro",
}


DIRECT_ADDRESS = {
    "tu",
    "usted",
    "debes",
    "deberias",
    "tienes que",
    "no te",
    "mira",
    "conoce",
    "descubre",
    "atencion",
}


LISTICLE_RE = re.compile(
    r"^\s*\d+\s+(cosas|razones|claves|formas|datos|tips|senales|errores|trucos|motivos)\b",
    flags=re.IGNORECASE,
)


VAGUE_START_RE = re.compile(
    r"^\s*(esto|este|esta|estos|estas|asi|aqui|ella|el|ellos|ellas)\b",
    flags=re.IGNORECASE,
)


QUESTION_START_RE = re.compile(r"^\s*(que|como|cuando|donde|por que|cuanto|cual|cuales|quien)\b", flags=re.IGNORECASE)


GENERIC_LISTICLE_RE = re.compile(r"^\s*(?:[2-9]|1\d|20)\s+\w+", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ClickbaitPrediction:
    label: str
    score: float
    reasons: list[str]
    needs_review: bool


def _contains_any(text: str, vocabulary: set[str]) -> list[str]:
    hits = []
    for term in vocabulary:
        if re.search(rf"\b{re.escape(term)}\b", text):
            hits.append(term)
    return sorted(hits)


def score_clickbait(headline: str) -> tuple[float, list[str]]:
    raw = clean_text(headline)
    text = normalize_for_matching(raw)
    reasons: list[str] = []
    score = 0.0

    sensational_hits = _contains_any(text, SENSATIONAL_WORDS)
    if sensational_hits:
        score += min(0.30, 0.10 + 0.05 * len(sensational_hits))
        reasons.append("sensational_language:" + ",".join(sensational_hits[:4]))

    curiosity_hits = _contains_any(text, CURIOSITY_PHRASES)
    if curiosity_hits:
        score += min(0.36, 0.20 + 0.06 * len(curiosity_hits))
        reasons.append("curiosity_gap:" + ",".join(curiosity_hits[:4]))
        if any(text.startswith(hit) for hit in curiosity_hits):
            score += 0.10
            reasons.append("curiosity_gap_opener")
        if ":" in raw:
            score += 0.10
            reasons.append("curiosity_colon_framing")

    framing_hits = _contains_any(text, CLICKBAIT_FRAMING_PHRASES)
    if framing_hits:
        score += min(0.20, 0.08 + 0.04 * len(framing_hits))
        reasons.append("clickbait_framing:" + ",".join(framing_hits[:4]))

    reaction_hits = _contains_any(text, REACTION_PHRASES)
    if reaction_hits:
        score += min(0.24, 0.12 + 0.04 * len(reaction_hits))
        reasons.append("reaction_hook:" + ",".join(reaction_hits[:4]))

    address_hits = _contains_any(text, DIRECT_ADDRESS)
    if address_hits:
        score += min(0.20, 0.08 + 0.04 * len(address_hits))
        reasons.append("direct_address:" + ",".join(address_hits[:4]))

    if LISTICLE_RE.search(text):
        score += 0.23
        reasons.append("listicle_format")
    elif GENERIC_LISTICLE_RE.search(text):
        score += 0.23
        reasons.append("numbered_list_format")

    if text.startswith("adios a ") or (text.startswith("ni ") and " ni " in text):
        score += 0.23
        reasons.append("contrast_hook")

    if VAGUE_START_RE.search(text):
        score += 0.10
        reasons.append("vague_subject")

    if "?" in raw or "¿" in raw:
        score += 0.11
        reasons.append("question_headline")
        if QUESTION_START_RE.search(text):
            score += 0.05
            reasons.append("question_opener")
    elif QUESTION_START_RE.search(text):
        score += 0.16
        reasons.append("question_opener_without_punctuation")

    if "!" in raw or "¡" in raw:
        score += 0.08
        reasons.append("exclamation")

    if re.search(r"([!?]){2,}", raw):
        score += 0.08
        reasons.append("multiple_punctuation")

    uppercase_letters = sum(1 for ch in raw if ch.isupper())
    letters = sum(1 for ch in raw if ch.isalpha())
    if letters and uppercase_letters / letters >= 0.45 and letters >= 12:
        score += 0.12
        reasons.append("high_uppercase_ratio")

    concrete_signals = 0
    if re.search(r"\b(19|20)\d{2}\b|\b\d{1,2}[:/.-]\d{1,2}\b", raw):
        concrete_signals += 1
    if re.search(r"\b(gobierno|congreso|senado|corte|ministerio|municipalidad|fiscalia|banco central)\b", text):
        concrete_signals += 1
    if ":" in raw and not curiosity_hits:
        concrete_signals += 1
    if concrete_signals >= 2 and score < 0.45:
        score -= 0.08
        reasons.append("concrete_informative_signals")

    score = max(0.0, min(1.0, round(score, 3)))
    return score, reasons


def classify_headline(
    headline: str,
    *,
    threshold: float = DEFAULT_CLICKBAIT_THRESHOLD,
    review_margin: float = 0.08,
) -> ClickbaitPrediction:
    score, reasons = score_clickbait(headline)
    label = "clickbait" if score >= threshold else "informativo"
    needs_review = abs(score - threshold) <= review_margin
    return ClickbaitPrediction(label=label, score=score, reasons=reasons, needs_review=needs_review)
