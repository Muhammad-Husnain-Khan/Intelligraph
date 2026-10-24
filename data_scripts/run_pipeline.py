"""Run all Phase 2 stages in order: ``python -m data_scripts.run_pipeline``."""
from __future__ import annotations

import argparse
import json
import sys

from . import (
    chunk_data,
    clean_data,
    collect_files,
    collect_web,
    config,
    download_papers,
    features,
    make_report_numbers,
    run_eda,
)
from .utils import PipelineError, get_logger

logger = get_logger("run_pipeline")


def build_parser() -> argparse.ArgumentParser:
    """CLI argument parser."""
    p = argparse.ArgumentParser(description="IntelliGraph Phase 2 pipeline")
    p.add_argument("--skip-web", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-files", action="store_true")
    p.add_argument("--skip-clean", action="store_true", help="also skips features and chunking")
    p.add_argument("--skip-chunk", action="store_true", help="skip generating chunks")
    p.add_argument("--skip-eda", action="store_true")
    p.add_argument("--keep-log", action="store_true", help="do not clear the collection log")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; returns a process exit code."""
    args = build_parser().parse_args(argv)
    config.ensure_dirs()
    if not args.keep_log:
        config.COLLECTION_LOG.write_text("", encoding="utf-8")
        logger.info("collection log cleared")

    try:
        for skip, name, stage in (
            (args.skip_web, "web", collect_web.run),
            (args.skip_download, "download", download_papers.run),
            (args.skip_files, "files", collect_files.run),
        ):
            if skip:
                continue
            try:
                stage()
            except PipelineError as exc:  # a collection stage may legitimately be empty
                logger.warning("stage '%s': %s", name, exc)
        if not args.skip_clean:
            clean_data.run()
            features.run()
            if not args.skip_chunk:
                chunk_data.run()
        if not args.skip_eda:
            run_eda.run()
        make_report_numbers.run()
    except PipelineError as exc:
        logger.error("%s", exc)
        return 1

    audit = json.loads(config.AUDIT_JSON.read_text(encoding="utf-8"))
    print("\n=== Cleaning audit ===")
    for key, value in audit.items():
        print(f"{key:>20}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
