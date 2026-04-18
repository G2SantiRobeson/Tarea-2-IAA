from __future__ import annotations

import json
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .text_utils import (
    clean_text,
    headline_key,
    looks_like_valid_headline,
    stable_id,
    strip_common_title_suffix,
)


DEFAULT_USER_AGENT = (
    "Tarea2IAA-ClickbaitDataset/1.0 "
    "(academic web scraping; contact: student@example.com)"
)


def build_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-CL,es;q=0.9,en;q=0.7",
        }
    )
    return session


def fetch_text(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 20,
    retries: int = 2,
    sleep_seconds: float = 0.0,
) -> str | None:
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 429 and attempt < retries:
                time.sleep(max(3.0, sleep_seconds * 2))
                continue
            response.raise_for_status()
            encoding = response.apparent_encoding or response.encoding or "utf-8"
            return response.content.decode(encoding, errors="replace")
        except requests.RequestException:
            if attempt >= retries:
                return None
            time.sleep(1.0 + attempt)
    return None


def polite_sleep(base_delay: float) -> None:
    if base_delay <= 0:
        return
    time.sleep(base_delay + random.uniform(0, base_delay * 0.25))


def parse_datetime(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        return parsedate_to_datetime(text).isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).isoformat()
    except ValueError:
        return text


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tag_text(tag) -> str:
    return clean_text(tag.get_text(" ", strip=True)) if tag else ""


def first_tag_text(parent, names: Iterable[str]) -> str:
    for name in names:
        tag = parent.find(name)
        if tag:
            return tag_text(tag)
    for tag in parent.find_all(True):
        tag_name = (tag.name or "").lower()
        if any(tag_name.endswith(name.lower()) for name in names):
            text = tag_text(tag)
            if text:
                return text
    return ""


def make_record(
    *,
    headline: str,
    url: str,
    source: dict,
    author: str = "",
    published_at: str = "",
    section: str = "",
    collection_method: str,
    extra: dict | None = None,
) -> dict:
    headline = strip_common_title_suffix(headline)
    url = clean_text(url)
    payload = {
        "id": stable_id(source["name"], url, headline),
        "headline": headline,
        "url": url,
        "source": source["name"],
        "source_type": source["source_type"],
        "author": clean_text(author),
        "published_at": parse_datetime(published_at),
        "section": clean_text(section),
        "collection_method": collection_method,
        "scraped_at": now_utc_iso(),
    }
    if extra:
        payload.update(extra)
    return payload


def discover_feed_urls(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    feeds: list[str] = []
    for link in soup.find_all("link"):
        rel = " ".join(link.get("rel", [])).lower()
        kind = clean_text(link.get("type")).lower()
        href = clean_text(link.get("href"))
        if href and "alternate" in rel and ("rss" in kind or "atom" in kind):
            feeds.append(urljoin(base_url, href))
    return sorted(set(feeds))


def candidate_feed_urls(source: dict, session: requests.Session) -> list[str]:
    base_url = source["base_url"].rstrip("/")
    candidates = list(source.get("rss_urls", []))
    homepage = fetch_text(session, base_url, timeout=15, retries=1)
    if homepage:
        candidates.extend(discover_feed_urls(homepage, base_url))
    candidates.extend([f"{base_url}/feed/", f"{base_url}/rss", f"{base_url}/rss.xml"])
    seen: set[str] = set()
    ordered: list[str] = []
    for url in candidates:
        if url and url not in seen:
            ordered.append(url)
            seen.add(url)
    return ordered


def parse_feed(xml_text: str, source: dict, feed_url: str) -> list[dict]:
    soup = BeautifulSoup(xml_text, "xml")
    records: list[dict] = []

    for item in soup.find_all("item"):
        headline = first_tag_text(item, ["title"])
        link = first_tag_text(item, ["link"]) or first_tag_text(item, ["guid"])
        if not looks_like_valid_headline(headline):
            continue
        records.append(
            make_record(
                headline=headline,
                url=link,
                source=source,
                author=first_tag_text(item, ["creator", "author", "dc:creator"]),
                published_at=first_tag_text(item, ["pubDate", "published", "updated"]),
                section=first_tag_text(item, ["category"]),
                collection_method="rss",
                extra={"feed_url": feed_url},
            )
        )

    for entry in soup.find_all("entry"):
        headline = first_tag_text(entry, ["title"])
        link_tag = entry.find("link")
        link = clean_text(link_tag.get("href")) if link_tag else ""
        link = link or first_tag_text(entry, ["id"])
        if not looks_like_valid_headline(headline):
            continue
        records.append(
            make_record(
                headline=headline,
                url=link,
                source=source,
                author=first_tag_text(entry, ["author", "name"]),
                published_at=first_tag_text(entry, ["published", "updated"]),
                section=first_tag_text(entry, ["category"]),
                collection_method="atom",
                extra={"feed_url": feed_url},
            )
        )
    return records


def collect_feed_records(source: dict, session: requests.Session) -> list[dict]:
    records: list[dict] = []
    for feed_url in candidate_feed_urls(source, session):
        xml_text = fetch_text(session, feed_url, timeout=15, retries=1)
        if not xml_text or "<" not in xml_text[:100]:
            continue
        records.extend(parse_feed(xml_text, source, feed_url))
    return records


def parse_json_ld(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    objects: list[dict] = []
    for script in soup.find_all("script"):
        kind = clean_text(script.get("type")).lower()
        if "ld+json" not in kind:
            continue
        raw = script.string or script.get_text(" ", strip=True)
        raw = clean_text(raw)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        objects.extend(flatten_json_ld(parsed))
    return objects


def flatten_json_ld(value) -> list[dict]:
    if isinstance(value, list):
        out: list[dict] = []
        for item in value:
            out.extend(flatten_json_ld(item))
        return out
    if not isinstance(value, dict):
        return []
    out = [value]
    graph = value.get("@graph")
    if isinstance(graph, list):
        for item in graph:
            out.extend(flatten_json_ld(item))
    return out


def jsonld_type_matches(obj: dict, expected: str) -> bool:
    kind = obj.get("@type")
    if isinstance(kind, list):
        values = [clean_text(item).lower() for item in kind]
    else:
        values = [clean_text(kind).lower()]
    expected = expected.lower()
    return any(expected == value or value.endswith(expected) for value in values)


def first_jsonld_value(objects: list[dict], keys: Iterable[str]) -> str:
    for obj in objects:
        if not any(jsonld_type_matches(obj, kind) for kind in ("NewsArticle", "Article", "ReportageNewsArticle")):
            continue
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str):
                return clean_text(value)
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str):
                    return clean_text(first)
                if isinstance(first, dict):
                    return clean_text(first.get("name"))
            if isinstance(value, dict):
                return clean_text(value.get("name"))
    return ""


def meta_content(soup: BeautifulSoup, selectors: Iterable[dict]) -> str:
    for attrs in selectors:
        tag = soup.find("meta", attrs=attrs)
        if tag:
            value = clean_text(tag.get("content"))
            if value:
                return value
    return ""


def extract_article_metadata(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    jsonld = parse_json_ld(html)

    headline = (
        meta_content(
            soup,
            [
                {"property": "og:title"},
                {"name": "twitter:title"},
                {"name": "title"},
            ],
        )
        or first_jsonld_value(jsonld, ["headline", "name"])
        or tag_text(soup.find("h1"))
        or tag_text(soup.find("title"))
    )
    author = (
        meta_content(
            soup,
            [
                {"name": "author"},
                {"property": "article:author"},
                {"name": "parsely-author"},
            ],
        )
        or first_jsonld_value(jsonld, ["author", "creator"])
    )
    published_at = (
        meta_content(
            soup,
            [
                {"property": "article:published_time"},
                {"name": "article:published_time"},
                {"name": "pubdate"},
                {"name": "date"},
                {"name": "dc.date"},
            ],
        )
        or first_jsonld_value(jsonld, ["datePublished", "dateCreated"])
    )
    section = (
        meta_content(
            soup,
            [
                {"property": "article:section"},
                {"name": "section"},
                {"name": "parsely-section"},
            ],
        )
        or first_jsonld_value(jsonld, ["articleSection"])
    )
    return {
        "headline": strip_common_title_suffix(headline),
        "url": url,
        "author": author,
        "published_at": published_at,
        "section": section,
    }


def parse_sitemap_locations(xml_text: str) -> tuple[list[str], bool]:
    soup = BeautifulSoup(xml_text, "xml")
    locs = [clean_text(tag.get_text(" ", strip=True)) for tag in soup.find_all("loc")]
    is_index = bool(soup.find("sitemapindex"))
    return [loc for loc in locs if loc], is_index


def discover_sitemap_urls_from_robots(session: requests.Session, base_url: str) -> list[str]:
    robots_url = f"{base_url.rstrip('/')}/robots.txt"
    text = fetch_text(session, robots_url, timeout=15, retries=1)
    if not text:
        return []
    urls: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.lower().startswith("sitemap:"):
            continue
        url = clean_text(line.split(":", 1)[1])
        if url:
            urls.append(url)
    return urls


def looks_like_article_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if not parsed.scheme.startswith("http"):
        return False
    bad_tokens = (
        "/tag/",
        "/tags/",
        "/category/",
        "/categorias/",
        "/author/",
        "/autores/",
        "/page/",
        "/wp-content/",
        "/static/",
        "/assets/",
        "/newsletter",
        "/rss",
        "/feed",
    )
    bad_suffixes = (".xml", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".pdf", ".mp4", ".css", ".js")
    if any(token in path for token in bad_tokens) or path.endswith(bad_suffixes):
        return False
    clean_path = path.strip("/")
    if not clean_path:
        return False
    path_parts = [part for part in clean_path.split("/") if part]
    return len(path_parts) >= 2 or any(ch.isdigit() for ch in clean_path)


def iter_article_urls_from_sitemaps(
    source: dict,
    session: requests.Session,
    *,
    max_sitemaps: int = 30,
    max_urls: int = 600,
) -> list[str]:
    base = source["base_url"].rstrip("/")
    initial_sitemaps = []
    initial_sitemaps.extend(source.get("sitemap_urls", []))
    initial_sitemaps.extend(discover_sitemap_urls_from_robots(session, base))
    initial_sitemaps.append(f"{base}/sitemap.xml")
    queue = deque(dict.fromkeys(url for url in initial_sitemaps if url))
    seen_sitemaps: set[str] = set()
    seen_urls: set[str] = set()
    article_urls: list[str] = []

    while queue and len(seen_sitemaps) < max_sitemaps and len(article_urls) < max_urls:
        sitemap_url = queue.popleft()
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)
        xml_text = fetch_text(session, sitemap_url, timeout=20, retries=1)
        if not xml_text:
            continue
        locs, is_index = parse_sitemap_locations(xml_text)
        for loc in locs:
            if len(article_urls) >= max_urls:
                break
            lower = loc.lower()
            if is_index or lower.endswith(".xml") or "sitemap" in lower:
                if loc not in seen_sitemaps and len(seen_sitemaps) + len(queue) < max_sitemaps:
                    queue.append(loc)
                continue
            if looks_like_article_url(loc) and loc not in seen_urls:
                seen_urls.add(loc)
                article_urls.append(loc)
    return article_urls


def collect_sitemap_records(
    source: dict,
    session: requests.Session,
    *,
    max_records: int,
    delay_seconds: float,
) -> list[dict]:
    records: list[dict] = []
    for url in iter_article_urls_from_sitemaps(source, session, max_urls=max_records * 3):
        if len(records) >= max_records:
            break
        html = fetch_text(session, url, timeout=20, retries=1)
        polite_sleep(delay_seconds)
        if not html:
            continue
        metadata = extract_article_metadata(html, url)
        headline = metadata.get("headline", "")
        if not looks_like_valid_headline(headline):
            continue
        records.append(
            make_record(
                headline=headline,
                url=url,
                source=source,
                author=metadata.get("author", ""),
                published_at=metadata.get("published_at", ""),
                section=metadata.get("section", ""),
                collection_method="sitemap_article",
            )
        )
    return records


def add_unique_record(records: list[dict], seen_keys: set[str], record: dict) -> bool:
    key = headline_key(record.get("headline", ""))
    if not key or key in seen_keys:
        return False
    seen_keys.add(key)
    records.append(record)
    return True


def scrape_sources(
    sources: list[dict],
    *,
    target_per_group: int = 1000,
    max_per_source: int = 350,
    delay_seconds: float = 0.8,
    include_sitemaps: bool = True,
) -> list[dict]:
    session = build_session()
    rows: list[dict] = []
    seen_keys: set[str] = set()
    group_counts: dict[str, int] = defaultdict(int)

    for source in sources:
        group = source["source_type"]
        if group_counts[group] >= target_per_group:
            continue

        per_source_added = 0
        for record in collect_feed_records(source, session):
            if group_counts[group] >= target_per_group or per_source_added >= max_per_source:
                break
            if add_unique_record(rows, seen_keys, record):
                group_counts[group] += 1
                per_source_added += 1

        if not include_sitemaps or group_counts[group] >= target_per_group:
            continue

        remaining_for_source = max_per_source - per_source_added
        if remaining_for_source <= 0:
            continue

        for record in collect_sitemap_records(
            source,
            session,
            max_records=remaining_for_source,
            delay_seconds=delay_seconds,
        ):
            if group_counts[group] >= target_per_group or per_source_added >= max_per_source:
                break
            if add_unique_record(rows, seen_keys, record):
                group_counts[group] += 1
                per_source_added += 1

    return rows


def extract_claimreview_metadata(html: str, url: str) -> dict:
    jsonld = parse_json_ld(html)
    claim_objects = [obj for obj in jsonld if jsonld_type_matches(obj, "ClaimReview")]
    if not claim_objects:
        article = extract_article_metadata(html, url)
        return {
            "claim": article.get("headline", ""),
            "rating": "",
            "review_title": article.get("headline", ""),
            "published_at": article.get("published_at", ""),
            "author": article.get("author", ""),
        }

    claim = claim_objects[0]
    rating = claim.get("reviewRating") or {}
    if isinstance(rating, list):
        rating = rating[0] if rating else {}
    item_reviewed = claim.get("itemReviewed") or {}
    if isinstance(item_reviewed, list):
        item_reviewed = item_reviewed[0] if item_reviewed else {}

    claim_text = (
        clean_text(claim.get("claimReviewed"))
        or clean_text(item_reviewed.get("name") if isinstance(item_reviewed, dict) else "")
        or clean_text(claim.get("name"))
    )
    rating_text = ""
    if isinstance(rating, dict):
        rating_text = (
            clean_text(rating.get("alternateName"))
            or clean_text(rating.get("name"))
            or clean_text(rating.get("ratingValue"))
        )
    return {
        "claim": claim_text,
        "rating": rating_text,
        "review_title": clean_text(claim.get("headline") or claim.get("name")),
        "published_at": clean_text(claim.get("datePublished") or claim.get("dateCreated")),
        "author": clean_text(claim.get("author", {}).get("name") if isinstance(claim.get("author"), dict) else claim.get("author")),
    }


def rating_suggests_falsehood(rating: str) -> bool:
    rating = clean_text(rating).lower()
    if not rating:
        return False
    false_tokens = (
        "false",
        "falso",
        "falsa",
        "fake",
        "incorrect",
        "incorrecto",
        "engaños",
        "enganos",
        "misleading",
        "deceptive",
        "pants on fire",
        "no evidence",
        "sin evidencia",
        "mostly false",
        "half false",
        "partly false",
    )
    return any(token in rating for token in false_tokens)


def scrape_fake_news_claims(
    sources: list[dict],
    *,
    target_total: int = 1000,
    max_per_source: int = 250,
    delay_seconds: float = 0.8,
    keep_unrated: bool = True,
) -> list[dict]:
    session = build_session()
    rows: list[dict] = []
    seen_keys: set[str] = set()

    for source in sources:
        if len(rows) >= target_total:
            break
        candidate_urls: list[str] = []
        for record in collect_feed_records(source, session):
            url = clean_text(record.get("url"))
            if url:
                candidate_urls.append(url)
        candidate_urls.extend(
            iter_article_urls_from_sitemaps(source, session, max_sitemaps=35, max_urls=max_per_source * 3)
        )

        source_added = 0
        for url in candidate_urls:
            if len(rows) >= target_total or source_added >= max_per_source:
                break
            html = fetch_text(session, url, timeout=20, retries=1)
            polite_sleep(delay_seconds)
            if not html:
                continue
            claim = extract_claimreview_metadata(html, url)
            headline = claim.get("claim", "")
            rating = claim.get("rating", "")
            if not looks_like_valid_headline(headline):
                continue
            if rating and not rating_suggests_falsehood(rating):
                continue
            if not rating and not keep_unrated:
                continue
            record = make_record(
                headline=headline,
                url=url,
                source=source,
                author=claim.get("author", ""),
                published_at=claim.get("published_at", ""),
                section="fact_check",
                collection_method="claimreview",
                extra={
                    "label": "fake_news",
                    "fact_check_rating": rating,
                    "fact_check_title": claim.get("review_title", ""),
                },
            )
            if add_unique_record(rows, seen_keys, record):
                source_added += 1

    return rows
