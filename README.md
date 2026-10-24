# IntelliGraph

**AI-powered chat with any website or document, grounded in a knowledge graph.**

IntelliGraph lets a user supply a website URL or upload a document (PDF, DOCX, TXT). The planned pipeline scrapes and cleans the text, builds a knowledge graph of the main concepts and their relations, and answers questions *only* from that data, with citations to the source passages. If the source holds no evidence, the system says so instead of guessing.

This repository currently contains **Deliverable 1** only:

| Phase | Content | Where |
|-------|---------|-------|
| 1. Problem Definition | Problem, audience, AI justification | `report/main.tex` |
| 2. Data Acquisition & Preprocessing | Collection, cleaning, feature strategy, EDA | `data_scripts/`, `report/main.tex` |

Later phases (retrieval, knowledge graph, conversational answering, web app) are **not** implemented here.

## Team

| Member | Roll No. |
|--------|----------|
| Muhammad Husnain Khan | 24L-3007 |
| Muhammad Muazzam Mahmood | 24L-3063 |
| Rai Muhammad Umais | 24L-3032 |

Repository: <https://github.com/rai-umais/IntelliGraph>

## Data

Theme: **AI and machine learning**, so a later knowledge graph has real links between pages.

* **Websites** (`data/manifests/urls.csv`): English Wikipedia articles, official scikit-learn / Python documentation, and a few Urdu Wikipedia pages (`expected_language=ur`, language labels are assigned manually in the manifest). Wikipedia text is CC BY-SA.
* **Files** (`data/raw/uploads/`, git-ignored): open-access arXiv PDFs listed in `data/manifests/papers.csv`, plus course notes written/owned by the team (DOCX and TXT) that you place in the same folder.
* **Languages:** English and Urdu. **Roman Urdu is not included** because no permitted example text was identified; the language heuristic also cannot detect it.
* No Kaggle/UCI datasets are used: the project retrieves from documents, not tabular data.
* Only public/permitted content is collected; `robots.txt` is honoured. Some failures (JavaScript-heavy pages, scanned PDFs, robots blocks) are expected, are written to `logs/collection_log.jsonl`, and are reported rather than hidden.

All manifest entries are a starting point; **verify them** (especially the Urdu URLs) before the real run.

## Folder structure

```
.
├── README.md
├── .gitignore  .env.example  requirements.txt  pytest.ini
├── data_scripts/          # ALL Phase 2 code
│   ├── config.py          # paths, thresholds, timeouts, User-Agent
│   ├── utils.py           # logging, JSONL, language heuristic, SSRF + robots.txt checks
│   ├── collect_web.py     # scrape URLs from the manifest
│   ├── download_papers.py # download arXiv PDFs into uploads/
│   ├── collect_files.py   # extract text from PDF/DOCX/TXT
│   ├── clean_data.py      # cleaning + audit
│   ├── features.py        # normalization, scaling, encoding
│   ├── run_eda.py         # 4 report figures
│   ├── make_report_numbers.py  # report/generated/numbers.tex
│   └── run_pipeline.py    # runs everything in order
├── data/
│   ├── manifests/         # urls.csv, papers.csv (committed)
│   └── raw/ interim/ processed/   # contents git-ignored
├── logs/                  # collection_log.jsonl, cleaning_audit.json
├── report/                # main.tex, figures/, generated/numbers.tex
└── tests/                 # offline tests
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

Record schema: `source_id, title, source_type (website|pdf|docx|txt), source_uri, collected_at (UTC), language (en|ur|roman_ur|unknown), text, metadata`.

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

## Reproducibility notes

* Deterministic processing: sorted file order, fixed thresholds (`MIN_WORDS=50`, `IQR_K=1.5`) in `data_scripts/config.py`.
* Web content changes over time; the collection timestamp is stored per record and the date range appears in the report.
* `INTELLIGRAPH_ROOT` overrides the repository root (used for tests/smoke runs).
* No secrets in code; contact email comes from `.env` (never committed).
* Duplicate removal keys on SHA-256 of `source_uri + normalized text`; identical text from different URIs is kept for citation provenance.
* Outliers (1.5×IQR on word count) are flagged and kept; flagging is skipped with fewer than 4 records.
