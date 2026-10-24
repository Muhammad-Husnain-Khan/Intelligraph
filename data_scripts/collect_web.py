"""Scrape the websites listed in ``data/manifests/urls.csv`` into ``web_records.jsonl``."""
from __future__ import annotations

import csv
import re
import time
from typing import Any

from bs4 import BeautifulSoup

from . import config
from .utils import (
    PipelineError,
    RobotsChecker,
    fetch_with_retries,
    get_logger,
    log_event,
    make_session,
    merge_jsonl,
    source_id,
    utc_now_iso,
)

logger = get_logger(__name__)

VALID_LANGS = {"en", "ur", "roman_ur"}
_REMOVE_TAGS = ["script", "style", "noscript", "nav", "footer", "aside"]
_REMOVE_SELECTORS = [
    "sup.reference",
    ".mw-editsection",
    ".navbox",
    ".vertical-navbox",
    ".reflist",
    "ol.references",
    ".mw-references-wrap",
    "#toc",
    ".toc",
]
_REGION_SELECTORS = ["#mw-content-text", "main", "article", "[role=main]"]
_TEXT_TAGS = ["h1", "h2", "h3", "h4", "p", "li"]


def read_manifest(path=None) -> list[dict[str, str]]:
    """Read urls.csv, skipping blank lines and lines starting with ``#``."""
    path = path or config.URLS_CSV
    if not path.exists():
        raise PipelineError(f"URL manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    rows = []
    for row in csv.DictReader(lines):
        url = (row.get("url") or "").strip()
        if url:
            rows.append(
                {
                    "url": url,
                    "expected_language": (row.get("expected_language") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return rows


def extract_text(html: str | bytes) -> tuple[str, str]:
    """Return ``(title, text)`` extracted from HTML.

    Boilerplate is removed; h1-h4, p and li texts from the main region are joined
    by blank lines so paragraphs survive.
    """
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(id="firstHeading") or soup.find("h1")
    title = heading.get_text(" ", strip=True) if heading else ""
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    for tag in soup(_REMOVE_TAGS):
        tag.decompose()
    for selector in _REMOVE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()

    region = None
    for selector in _REGION_SELECTORS:
        region = soup.select_one(selector)
        if region is not None:
            break
    region = region or soup.body or soup

    parts: list[str] = []
    for el in region.find_all(_TEXT_TAGS):
        if el.find_parent(["li", "p"]) is not None:
            continue  # text is already included by the enclosing li/p
        text = re.sub(r"\s+", " ", el.get_text()).strip()
        if text:
            parts.append(text)
    return title, "\n\n".join(parts)


def _is_html(content_type: str) -> bool:
    return any(t in content_type.lower() for t in ("text/html", "application/xhtml"))


def collect_one(
    session, robots: RobotsChecker, row: dict[str, str]
) -> dict[str, Any] | None:
    """Collect a single URL; logs the attempt and returns a record or None."""
    url = row["url"]
    allowed, why = robots.allowed(url)
    if not allowed:
        log_event("web", url, False, None, f"skipped: {why}")
        logger.warning("skip %s (%s)", url, why)
        return None

    resp, code, reason = fetch_with_retries(session, url)
    if resp is None or not resp.ok:
        log_event("web", url, False, code, reason)
        logger.warning("fail %s (%s)", url, reason)
        return None

    ctype = resp.headers.get("Content-Type", "")
    if not _is_html(ctype):
        log_event("web", url, False, code, f"not HTML: {ctype or 'no content-type'}")
        return None

    title, text = extract_text(resp.content)
    if not text.strip():
        log_event("web", url, False, code, "no extractable text (JS-rendered page?)")
        return None

    lang = row["expected_language"] if row["expected_language"] in VALID_LANGS else "unknown"
    log_event("web", url, True, code, "ok")
    return {
        "source_id": source_id(url),
        "title": title,
        "source_type": "website",
        "source_uri": url,
        "collected_at": utc_now_iso(),
        "language": lang,
        "text": text,
        "metadata": {
            "http_code": code,
            "final_url": resp.url,
            "content_type": ctype,
            "language_source": "manifest (manual)",
        },
    }


def run() -> int:
    """Scrape every manifest URL. Returns the number of successful records."""
    config.ensure_dirs()
    rows = read_manifest()
    if not rows:
        raise PipelineError(f"No URLs in manifest: {config.URLS_CSV}")
    session = make_session()
    robots = RobotsChecker(session)
    records = []
    for i, row in enumerate(rows):
        logger.info("[%d/%d] %s", i + 1, len(rows), row["url"])
        rec = collect_one(session, robots, row)
        if rec:
            records.append(rec)
        if i < len(rows) - 1:
            time.sleep(config.REQUEST_DELAY)
    added, replaced, total = merge_jsonl(config.WEB_RECORDS, records)
    logger.info(
        "web: %d/%d succeeded (%d new, %d replaced); %d total records in %s",
        len(records), len(rows), added, replaced, total, config.WEB_RECORDS,
    )
    return len(records)


if __name__ == "__main__":
    run()
