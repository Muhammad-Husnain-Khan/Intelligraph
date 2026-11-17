# IntelliGraph

**AI-powered chat with any website or document, grounded in a knowledge graph.**

IntelliGraph lets a user supply a website URL or upload a document (PDF, DOCX, TXT). The planned pipeline scrapes and cleans the text, builds a knowledge graph of the main concepts and their relations, and answers questions *only* from that data, with citations to the source passages. If the source holds no evidence, the system says so instead of guessing.

This repository covers all three project deliverables:

| Deliverable | Phases & Content | Where |
|-------------|------------------|-------|
| 1 | Phases 1–2: Problem definition, collection, cleaning, chunking, EDA | `data_scripts/`, `report/main.tex` |
| 2 | Phases 3–4: Vector, Graph & Hybrid retrieval, hyperparameter sweeps, 48-Q eval | `retrieval/`, `tuning/`, `eval/`, `report/main2.tex` |
| 3 (Track A) | Phases 5–6: Runtime ingestion, SSE streaming API, React Reading Room UI, Docker | `api/`, `ingest/`, `web/`, `scripts/`, `report/main3.tex` |

## Run

**Docker (whole stack):** `docker compose up --build`, then open http://localhost:8080.

**Local:**
```bash
docker start pg-vector neo4j          # or the docker run lines in the Deliverable 2 section
pip install -r requirements.txt
cp .env.example .env                  # then add your own key; .env is git-ignored
python -m scripts.check_llm           # confirms a real model + one round trip
python -m scripts.seed_library        # loads data/processed/chunks.jsonl as the library (if it exists)
python -m uvicorn api.main:app --port 8000
cd web && npm install && npm run dev  # http://localhost:5173
pytest tests/test_track_a_backend.py  # mocked tests; the older tests need live Postgres/Neo4j
```

| Variable | Purpose |
|----------|---------|
| `INTELLIGRAPH_GEMINI_API_KEY` (or `GEMINI_API_KEY`) | Google AI Studio key (primary LLM) |
| `GROQ_API_KEY` | Optional OpenAI-compatible backup |
| `INTELLIGRAPH_LLM_PROVIDER` | `gemini` (default) or `openai_compat` |
| `INTELLIGRAPH_GEMINI_MODEL` / `..._MODEL_PREFERENCE` | Optional pin / preference list; default auto-picks from the live model list |
| `INTELLIGRAPH_PGVECTOR_DSN`, `INTELLIGRAPH_NEO4J_*` | Store connections |
| `INTELLIGRAPH_MAX_UPLOAD_MB`, `INTELLIGRAPH_FRONTEND_ORIGIN`, `INTELLIGRAPH_RATE_LIMIT_PER_MIN` | Upload limit, CORS origins, per-IP rate limit |

API: `GET /api/health`, `GET/DELETE /api/sources`, `POST /api/sources/{files,url,text}`, `GET /api/sources/{id}/events` (SSE), `POST /api/chat/stream` (SSE), `POST /api/chat`, `POST /api/compare`, `GET /api/graph`. Legacy `/health` and `/query/*` are unchanged.

## Team

| Member | Roll No. |
|--------|----------|
| Muhammad Husnain Khan | 24L-3007 |
| Muhammad Muazzam Mahmood | 24L-3063 |
| Rai Muhammad Umais | 24L-3032 |

Repository: <https://github.com/Muhammad-Husnain-Khan/IntelliGraph>

## Data

Theme: **AI and machine learning**, so a later knowledge graph has real links between pages.

* **Websites** (`data/manifests/urls.csv`): English Wikipedia articles, official scikit-learn / Python documentation, telecommunications portals, software architecture articles, and culinary guides. Wikipedia text is CC BY-SA.
* **Files** (`data/raw/uploads/`, git-ignored): open-access arXiv PDFs listed in `data/manifests/papers.csv`, plus course notes written/owned by the team (DOCX and TXT) that you place in the same folder.
* **Languages:** English.
* No Kaggle/UCI datasets are used: the project retrieves from documents, not tabular data.
* Only public/permitted content is collected; `robots.txt` is honoured. Some failures (JavaScript-heavy pages, scanned PDFs, robots blocks) are expected, are written to `logs/collection_log.jsonl`, and are reported rather than hidden.

All manifest entries are a starting point; **verify them** before the real run.

## Folder structure

```text
.
├── README.md
├── .gitignore  .env.example  requirements.txt  pytest.ini  pyrightconfig.json
├── Dockerfile                 # FastAPI backend container
├── docker-compose.yml         # Full stack: pgvector, neo4j, api, web (nginx)
├── api/
│   └── main.py                # FastAPI service (/health, /query/*, /api/* SSE & REST)
├── ingest/
│   └── pipeline.py            # Runtime file/URL/text ingestion + per-stage SSE events
├── retrieval/
│   ├── embed.py               # Local SentenceTransformer (all-MiniLM-L6-v2)
│   ├── store.py               # Pooled Postgres/pgvector + Neo4j storage & upserts
│   ├── llm.py                 # Gemini & OpenAI-compat (Groq) provider layer + probe
│   ├── gemini_client.py       # Legacy REST client for /query/* endpoints
│   ├── chat.py                # Streaming hybrid chat, bridge passages, citation check
│   ├── index_chunks.py        # Batch pgvector indexer (Deliverable 2 CLI)
│   ├── graph_build.py         # Sentence co-occurrence entity graph builder
│   ├── vector_retriever.py    # Architecture 1: Vector-only retriever
│   ├── graph_retriever.py     # Architecture 2: Graph-only retriever
│   └── hybrid_retriever.py    # Architecture 3: Hybrid retriever
├── web/                       # React + Vite + TypeScript frontend (Reading Room UI)
│   ├── Dockerfile  nginx.conf # Production static build + unbuffered /api SSE proxy
│   ├── package.json  tsconfig.json  vite.config.ts  index.html
│   └── src/
│       ├── main.tsx           # App entry + self-hosted Newsreader & IBM Plex Mono fonts
│       ├── App.tsx            # Transcript, User/Developer modes, Thinking depth, Evidence
│       ├── Library.tsx        # Source upload, URL/text ingestion, scope & disclosures
│       ├── api.ts             # SSE & REST client
│       ├── config.ts          # Shared Thinking levels (K=3/5/10), heading & title helpers
│       ├── types.ts           # TypeScript contracts
│       └── styles.css         # Midnight Ink & Warm Ivory design tokens + print rules
├── data_scripts/              # Phase 2 data pipeline & heading extraction
│   ├── config.py              # Paths, thresholds, timeouts, runtime env settings
│   ├── utils.py               # Logging, JSONL, language heuristic, SSRF + robots.txt
│   ├── collect_web.py         # Scrape URLs from manifest
│   ├── download_papers.py     # Download arXiv PDFs into uploads/
│   ├── collect_files.py       # Extract text & headings from PDF/DOCX/TXT
│   ├── clean_data.py          # Cleaning + audit
│   ├── chunk_data.py          # Heading-aware 300/50 hybrid chunker
│   ├── features.py            # Normalization, scaling, encoding
│   ├── run_eda.py             # EDA report figures
│   ├── make_report_numbers.py # Writes report/generated/numbers.tex
│   └── run_pipeline.py        # Batch pipeline runner
├── scripts/
│   ├── check_llm.py           # Pre-demo LLM probe & round-trip check
│   ├── seed_library.py        # Incremental loader for data/processed/chunks.jsonl
│   └── contrast.py            # WCAG AA colour-contrast verification
├── eval/                      # Phase 4 golden QA benchmark (48 Qs), metrics & bias check
├── tuning/                    # Chunking & embedding model sweeps
├── data/
│   ├── manifests/             # urls.csv, papers.csv (committed)
│   ├── sample_microservices.txt # Sample document for live upload testing
│   └── raw/ interim/ processed/ # Working data (contents git-ignored)
├── logs/                      # collection_log.jsonl, cleaning_audit.json
├── report/                    # main.tex, main2.tex, main3.tex, figures/, generated/
└── tests/                     # Unit & contract tests (offline + live DB integration)
```

## Setup

Python 3.10+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then set INTELLIGRAPH_CONTACT to your email (used in the User-Agent)
```

## Run

```bash
# 1. put your own course notes (.docx / .txt) in data/raw/uploads/  (optional but recommended)
# 2. full pipeline: web -> download -> files -> clean -> features -> eda -> report numbers
python -m data_scripts.run_pipeline
```

Flags: `--skip-web --skip-download --skip-files --skip-clean --skip-eda --keep-log`.
Raw records accumulate across runs: `web_records.jsonl` and `file_records.jsonl` are merged by `source_uri`. A URL/file collected again replaces its old record; everything else is kept, so you can test a few URLs at a time and keep earlier data. Cleaned outputs (clean records, features, audit, figures, chunks) are rebuilt from all accumulated raw records. The collection log is the exception: it is cleared at the start of each pipeline run (unless `--keep-log`), so it, and the log-based report numbers, describe only the latest run. To start from scratch, delete the files in `data/raw/`.

The collection log is cleared at the start unless `--keep-log` is given, so if you skip a collection stage, use `--keep-log` to retain its earlier entries.

If a stage is blocked (e.g. arXiv robots.txt), download the PDFs by hand into `data/raw/uploads/` named `<arxiv_id>.pdf`; they are picked up by `collect_files.py`.

Tests (offline): `python -m pytest`

## Outputs

| Output | Path |
|--------|------|
| Raw records | `data/raw/web_records.jsonl`, `data/raw/file_records.jsonl` |
| Cleaned records | `data/processed/clean_records.jsonl` |
| Scaled/encoded features (inspection only; no model is trained) | `data/processed/features.csv` |
| Collection log | `logs/collection_log.jsonl` |
| Cleaning audit | `logs/cleaning_audit.json` |
| Figures | `report/figures/{source_type_distribution,word_count_distribution,feature_correlation,language_distribution}.png` |
| Report numbers | `report/generated/numbers.tex` |

Record schema: `source_id, title, source_type (website|pdf|docx|txt), source_uri, collected_at (UTC), language (en|unknown), text, metadata`.

## Adding the results to the report

```bash
python -m data_scripts.run_pipeline
cd report
pdflatex main.tex
pdflatex main.tex     # second pass resolves references
```

Table values and observed counts fill in automatically. Red `[FILL: ...]` markers (abstract, interpretations, department/track, conclusion, etc.) must be written by hand. Uses only standard TeX Live packages.

## Chunking (Phase 2 follow-on)

`data_scripts/chunk_data.py` splits the cleaned records into retrieval chunks with a hybrid strategy: topic-wise first (headings and paragraphs, websites only; References/External links sections are dropped), then size-capped (about 300 words, 50-word sentence-aligned overlap, chunks under 50 words merged into a neighbour). Each chunk keeps `source_id`, `source_uri`, `title`, `section_heading`, `chunk_index`, word offsets, and a content hash.

Chunking runs automatically when executing `python -m data_scripts.run_pipeline`. To run it standalone:

```bash
python -m data_scripts.chunk_data      # writes data/processed/chunks.jsonl (git-ignored)
```

Heading detection is a heuristic (short line without sentence punctuation), so navigation-heavy pages produce noisy headings. Chunk size is a baseline to be compared with Recall@K / MRR later.

## Deliverable 2: Retrieval, Generation and Evaluation (Phases 3-4)

Three architectures answer questions from `chunks.jsonl`:

| # | Architecture | Module |
|---|--------------|--------|
| 1 | Vector-only (pgvector, cosine) | `retrieval/vector_retriever.py` |
| 2 | Graph-only (Neo4j traversal) | `retrieval/graph_retriever.py` |
| 3 | Hybrid (text + graph facts in one prompt) | `retrieval/hybrid_retriever.py` |

Other parts: `retrieval/embed.py` (sentence-transformers), `retrieval/index_chunks.py` (loads Postgres), `retrieval/graph_build.py` (rule-based entity co-occurrence graph), `retrieval/gemini_client.py` (REST), `api/main.py` (FastAPI), `tuning/` (chunk-size and embedding sweeps), `eval/` (golden set, metrics, domain bias check).

```bash
docker run -d --name pg-vector -e POSTGRES_PASSWORD=pass -p 5433:5432 ankane/pgvector
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5
pip install -r requirements.txt           # then copy new variables from .env.example into .env
python -m retrieval.index_chunks          # embed + load pgvector
python -m retrieval.graph_build           # build Neo4j graph
uvicorn api.main:app                      # POST /query/{vector,graph,hybrid}  {"question": "..."}
python -m eval.run_eval [--generate]      # --generate needs INTELLIGRAPH_GEMINI_API_KEY
python -m eval.bias_check
python -m tuning.sweep_chunking ; python -m tuning.sweep_embeddings
```

Notes: Postgres is mapped to host port **5433** to avoid clashing with a local Postgres. Metrics use answer-string matching for relevance (a chunk is relevant if it contains an answer string). `eval/golden_qa.json` contains 48 verified English Q&A pairs covering diverse domains from the collected corpus. AUC-ROC is reported for vector and hybrid architectures; it is not defined for the graph architecture (no similarity score). Evaluation across $K=3, 5, 10$ establishes the Hybrid architecture with $K=5$ as the optimal production operating point (92.7% recall, 0.729 MRR). Results live in `eval/metrics_summary.json`, `eval/k_sweep_summary.json`, `eval/bias_report.json`, `tuning/results_*.csv`.

## Reproducibility notes

* Deterministic processing: sorted file order, fixed thresholds (`MIN_WORDS=50`, `IQR_K=1.5`) in `data_scripts/config.py`.
* Web content changes over time; the collection timestamp is stored per record and the date range appears in the report.
* `INTELLIGRAPH_ROOT` overrides the repository root (used for tests/smoke runs).
* No secrets in code; contact email comes from `.env` (never committed).
* Duplicate removal keys on SHA-256 of `source_uri + normalized text`; identical text from different URIs is kept for citation provenance.
* Outliers (1.5×IQR on word count) are flagged and kept; flagging is skipped with fewer than 4 records.
