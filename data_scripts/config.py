"""Central configuration for the IntelliGraph Phase 2 data pipeline.

All filesystem paths hang off ``ROOT``, which defaults to the repository root and can
be overridden with the ``INTELLIGRAPH_ROOT`` environment variable (used by tests and
smoke runs on temporary directories).
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT: Path = Path(
    os.environ.get("INTELLIGRAPH_ROOT", Path(__file__).resolve().parents[1])
).resolve()

# --- directories -----------------------------------------------------------
DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RAW_DIR = DATA_DIR / "raw"
UPLOADS_DIR = RAW_DIR / "uploads"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
GENERATED_DIR = REPORT_DIR / "generated"

# --- files -----------------------------------------------------------------
URLS_CSV = MANIFEST_DIR / "urls.csv"
PAPERS_CSV = MANIFEST_DIR / "papers.csv"
WEB_RECORDS = RAW_DIR / "web_records.jsonl"
FILE_RECORDS = RAW_DIR / "file_records.jsonl"
CLEAN_RECORDS = PROCESSED_DIR / "clean_records.jsonl"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
COLLECTION_LOG = LOGS_DIR / "collection_log.jsonl"
AUDIT_JSON = LOGS_DIR / "cleaning_audit.json"
NUMBERS_TEX = GENERATED_DIR / "numbers.tex"

# --- cleaning --------------------------------------------------------------
MIN_WORDS: int = 50
IQR_K: float = 1.5
MIN_RECORDS_FOR_OUTLIERS: int = 4

# --- collection ------------------------------------------------------------
REQUEST_TIMEOUT: float = 20.0
MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0  # seconds; sleeps 2, 4, 8 ...
MAX_RETRY_AFTER: float = 30.0
REQUEST_DELAY: float = 2.0  # polite delay between web requests
PAPER_DELAY: float = 3.0  # delay between arXiv downloads
MAX_REDIRECTS: int = 5
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# --- file extraction -------------------------------------------------------
SCANNED_MIN_CHARS_PER_PAGE: int = 50  # below this a PDF is treated as scanned
ARABIC_RATIO_THRESHOLD: float = 0.3

# --- identity --------------------------------------------------------------
CONTACT: str = os.environ.get("INTELLIGRAPH_CONTACT", "contact-not-set")
REPO_URL = "https://github.com/rai-umais/IntelliGraph"
USER_AGENT: str = (
    f"IntelliGraphBot/1.0 (university course project; +{REPO_URL}; {CONTACT})"
)


def ensure_dirs() -> None:
    """Create all working directories if they do not exist."""
    for directory in (
        MANIFEST_DIR,
        UPLOADS_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        LOGS_DIR,
        FIGURES_DIR,
        GENERATED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --- Phase 3/4: retrieval, generation, web app ------------------------------
# Secrets are read from the environment only (never from files in the repo).
def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
PGVECTOR_DSN: str = _env("INTELLIGRAPH_PGVECTOR_DSN", default="postgresql://postgres:pass@localhost:5433/postgres")
NEO4J_URI: str = _env("INTELLIGRAPH_NEO4J_URI", default="bolt://localhost:7687")
NEO4J_USER: str = _env("INTELLIGRAPH_NEO4J_USER", default="neo4j")
NEO4J_PASSWORD: str = _env("INTELLIGRAPH_NEO4J_PASSWORD", default="password")
GEMINI_API_KEY: str = _env("INTELLIGRAPH_GEMINI_API_KEY", "GEMINI_API_KEY")
GROQ_API_KEY: str = _env("INTELLIGRAPH_OPENAI_COMPAT_API_KEY", "GROQ_API_KEY")
OPENAI_COMPAT_BASE_URL: str = _env("INTELLIGRAPH_OPENAI_COMPAT_BASE_URL", default="https://api.groq.com/openai/v1")
OPENAI_COMPAT_MODEL: str = _env("INTELLIGRAPH_OPENAI_COMPAT_MODEL", default="llama-3.1-8b-instant")
LLM_PROVIDER: str = _env("INTELLIGRAPH_LLM_PROVIDER", default="gemini")
GEMINI_MODEL_PIN: str = _env("INTELLIGRAPH_GEMINI_MODEL")
GEMINI_MODEL_PREFERENCE: str = _env("INTELLIGRAPH_GEMINI_MODEL_PREFERENCE")
EMBEDDING_MODEL: str = _env("INTELLIGRAPH_EMBEDDING_MODEL", default="all-MiniLM-L6-v2")
TOP_K: int = int(_env("INTELLIGRAPH_TOP_K", default="5"))
MAX_UPLOAD_MB: int = int(_env("INTELLIGRAPH_MAX_UPLOAD_MB", default="20"))
FRONTEND_ORIGIN: str = _env("INTELLIGRAPH_FRONTEND_ORIGIN", default="http://localhost:5173,http://localhost:8080")
RATE_LIMIT_PER_MIN: int = int(_env("INTELLIGRAPH_RATE_LIMIT_PER_MIN", default="120"))
ANSWER_CACHE: bool = _env("INTELLIGRAPH_ANSWER_CACHE", default="0") == "1"
