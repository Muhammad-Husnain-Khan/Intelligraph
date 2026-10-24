"""Download the open-access arXiv PDFs listed in ``data/manifests/papers.csv``."""
from __future__ import annotations

import csv
import time

from . import config
from .utils import (
    PipelineError,
    RobotsChecker,
    fetch_with_retries,
    get_logger,
    log_event,
    make_session,
)

logger = get_logger(__name__)

ARXIV_PDF = "https://export.arxiv.org/pdf/{id}"


def read_papers(path=None) -> list[dict[str, str]]:
    """Read papers.csv (``arxiv_id, title``)."""
    path = path or config.PAPERS_CSV
    if not path.exists():
        raise PipelineError(f"Paper manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    return [
        {"arxiv_id": r["arxiv_id"].strip(), "title": (r.get("title") or "").strip()}
        for r in csv.DictReader(lines)
        if (r.get("arxiv_id") or "").strip()
    ]


def run() -> int:
    """Download missing PDFs into uploads/. Returns the number of PDFs now present."""
    config.ensure_dirs()
    papers = read_papers()
    session = make_session()
    robots = RobotsChecker(session)
    present = 0
    downloaded_any = False
    for paper in papers:
        pid = paper["arxiv_id"]
        target = config.UPLOADS_DIR / f"{pid.replace('/', '_')}.pdf"
        if target.exists():
            log_event("download", pid, True, None, "exists: skipped")
            present += 1
            continue
        url = ARXIV_PDF.format(id=pid)
        allowed, why = robots.allowed(url)
        if not allowed:
            log_event("download", pid, False, None, f"{why}; download manually")
            logger.warning("%s blocked: place %s by hand", pid, target.name)
            continue
        if downloaded_any:
            time.sleep(config.PAPER_DELAY)
        downloaded_any = True
        resp, code, reason = fetch_with_retries(session, url)
        if resp is None or not resp.ok:
            log_event("download", pid, False, code, f"{reason}; download manually")
            continue
        if not resp.content.startswith(b"%PDF"):
            log_event("download", pid, False, code, "response is not a PDF; download manually")
            continue
        target.write_bytes(resp.content)
        log_event("download", pid, True, code, "ok")
        present += 1
        logger.info("downloaded %s", target.name)
    logger.info("papers: %d/%d present in %s", present, len(papers), config.UPLOADS_DIR)
    return present


if __name__ == "__main__":
    run()
