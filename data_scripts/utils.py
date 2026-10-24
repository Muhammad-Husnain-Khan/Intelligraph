"""Shared helpers: logging, JSONL I/O, time, language heuristic, safe HTTP fetching."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import socket
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests

from . import config


class PipelineError(RuntimeError):
    """Raised for clear, user-facing pipeline failures (e.g. no data present)."""


class UnsafeURLError(ValueError):
    """Raised when a URL fails the SSRF check."""


# --------------------------------------------------------------------------- logging
def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single console handler."""
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- time / ids
def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def source_id(uri: str) -> str:
    """Stable 12-hex-char identifier: ``sha1(uri)[:12]``."""
    return hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]  # noqa: S324 (not security)


# --------------------------------------------------------------------------- JSONL
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file; returns an empty list if the file does not exist."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows to a JSONL file (overwrite). Returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def merge_jsonl(
    path: Path, new_rows: Iterable[dict[str, Any]], key: str = "source_uri"
) -> tuple[int, int, int]:
    """Append new rows to a JSONL file without discarding older data.

    A new row replaces an existing row with the same ``key``; all other existing rows
    are kept in their original order and new keys are added at the end.
    Returns ``(added, replaced, total)``.
    """
    merged: dict[str, dict[str, Any]] = {r[key]: r for r in read_jsonl(path) if key in r}
    added = replaced = 0
    for row in new_rows:
        if row[key] in merged:
            replaced += 1
        else:
            added += 1
        merged[row[key]] = row
    write_jsonl(path, merged.values())
    return added, replaced, len(merged)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append a single row to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_event(
    kind: str,
    source: str,
    success: bool,
    http_code: int | None = None,
    reason: str = "",
) -> None:
    """Append one attempt to ``logs/collection_log.jsonl``."""
    append_jsonl(
        config.COLLECTION_LOG,
        {
            "ts": utc_now_iso(),
            "kind": kind,
            "source": source,
            "success": success,
            "http_code": http_code,
            "reason": reason,
        },
    )


# --------------------------------------------------------------------------- language
def _is_arabic_script(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x0600 <= cp <= 0x06FF
        or 0x0750 <= cp <= 0x077F
        or 0xFB50 <= cp <= 0xFDFF
        or 0xFE70 <= cp <= 0xFEFF
    )


def arabic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Arabic-script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(_is_arabic_script(c) for c in letters) / len(letters)


def detect_language(text: str) -> str:
    """Heuristic: Arabic-script ratio > 0.3 -> ``ur`` else ``en``.

    This cannot detect Roman Urdu (Latin script); it is a script heuristic only.
    Returns ``unknown`` when the text has no letters.
    """
    if not any(c.isalpha() for c in text):
        return "unknown"
    return "ur" if arabic_ratio(text) > config.ARABIC_RATIO_THRESHOLD else "en"


# --------------------------------------------------------------------------- SSRF
def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_safe_url(url: str) -> tuple[bool, str]:
    """SSRF check: http/https only; every resolved IP must be a public address."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "unparseable URL"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme not allowed: {parsed.scheme or 'none'}"
    host = parsed.hostname
    if not host:
        return False, "missing host"
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, ValueError, UnicodeError) as exc:
        return False, f"cannot resolve host: {exc}"
    if not infos:
        return False, "host did not resolve"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if _ip_is_blocked(ip):
            return False, f"blocked non-public address: {ip}"
    return True, "ok"


def safe_get(
    session: requests.Session, url: str, *, timeout: float | None = None
) -> requests.Response:
    """GET with the SSRF guard applied to every hop; redirects are followed manually.

    Note: resolution is checked before the request, so a DNS-rebinding race remains
    theoretically possible; acceptable for this offline course pipeline.
    """
    timeout = timeout or config.REQUEST_TIMEOUT
    current = url
    for _ in range(config.MAX_REDIRECTS + 1):
        ok, why = is_safe_url(current)
        if not ok:
            raise UnsafeURLError(why)
        resp = session.get(current, timeout=timeout, allow_redirects=False)
        if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
            current = urljoin(current, resp.headers["Location"])
            continue
        return resp
    raise requests.TooManyRedirects(f"more than {config.MAX_REDIRECTS} redirects")


def make_session() -> requests.Session:
    """Session with the descriptive User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT, "Accept-Language": "en,ur;q=0.8"})
    return session


def fetch_with_retries(
    session: requests.Session, url: str
) -> tuple[requests.Response | None, int | None, str]:
    """Fetch with retry/backoff on 429, 5xx and connection errors.

    Returns ``(response_or_None, http_code_or_None, reason)``. A non-retryable
    response (e.g. 404) is returned as-is for the caller to judge.
    """
    reason = "unknown error"
    code: int | None = None
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            resp = safe_get(session, url)
        except UnsafeURLError as exc:
            return None, None, f"SSRF guard: {exc}"
        except requests.RequestException as exc:
            reason = f"{type(exc).__name__}: {exc}"[:200]
            code = None
        else:
            code = resp.status_code
            if code not in config.RETRY_STATUS:
                return resp, code, "ok" if resp.ok else f"HTTP {code}"
            reason = f"HTTP {code}"
            retry_after = resp.headers.get("Retry-After", "")
            if retry_after.isdigit():
                time.sleep(min(float(retry_after), config.MAX_RETRY_AFTER))
        if attempt < config.MAX_RETRIES:
            time.sleep(config.BACKOFF_BASE ** attempt)
    return None, code, f"gave up after {config.MAX_RETRIES} attempts: {reason}"


# --------------------------------------------------------------------------- robots.txt
class RobotsChecker:
    """Cached robots.txt checks. 4xx = allow, 5xx or unreachable = disallow."""

    def __init__(self, session: requests.Session) -> None:
        self.session = session
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def _load(self, origin: str) -> urllib.robotparser.RobotFileParser:
        parser = urllib.robotparser.RobotFileParser()
        robots_url = origin + "/robots.txt"
        try:
            resp = safe_get(self.session, robots_url)
        except (UnsafeURLError, requests.RequestException):
            parser.disallow_all = True
            return parser
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
        elif 400 <= resp.status_code < 500:
            parser.allow_all = True
        else:
            parser.disallow_all = True
        return parser

    def allowed(self, url: str) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for fetching ``url``."""
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        parser = self._cache[origin]
        if parser.can_fetch(config.USER_AGENT, url):
            return True, "robots.txt allows"
        return False, "disallowed by robots.txt (or robots.txt unavailable)"
