"""Offline tests for the cleaning stage (tiny in-memory synthetic strings only)."""
from data_scripts import config
from data_scripts.clean_data import (
    clean_records,
    compute_counts,
    iqr_fences,
    normalize_text,
)
from data_scripts.utils import detect_language, source_id


def _rec(uri: str, text: str, **extra):
    base = {"source_id": source_id(uri), "source_uri": uri, "text": text,
            "title": "T", "language": "en", "source_type": "txt"}
    base.update(extra)
    return base


LONG = "word " * 60


def test_normalize_text_keeps_paragraphs_and_urdu():
    out = normalize_text("A\x00  b\t\tc\r\n\r\n\r\n\r\nمصنوعی  ذہانت")
    assert out == "A b c\n\nمصنوعی ذہانت"


def test_missing_required_rejected():
    kept, audit = clean_records([_rec("a", LONG), _rec("b", ""), {"source_id": "x", "text": LONG}])
    assert audit["raw"] == 3 and audit["rejected_missing"] == 2 and audit["final"] == 1


def test_imputation_counts():
    kept, audit = clean_records([_rec("a", LONG, title="", language=None)])
    assert audit["imputed_title"] == 1 and audit["imputed_language"] == 1
    assert kept[0]["title"] == "a" and kept[0]["language"] == "unknown"


def test_duplicates_same_uri_removed_but_different_uri_kept():
    recs = [_rec("a", LONG), _rec("a", LONG), _rec("b", LONG)]
    kept, audit = clean_records(recs)
    assert audit["removed_duplicates"] == 1 and audit["final"] == 2


def test_short_records_excluded():
    kept, audit = clean_records([_rec("a", "too short"), _rec("b", LONG)])
    assert audit["excluded_short"] == 1 and audit["final"] == 1


def test_counts_include_urdu_full_stop():
    c = compute_counts("One. Two! Three?\n\nچار۔ پانچ۔")
    assert c["sentence_count"] == 5 and c["paragraph_count"] == 2


def test_outliers_flagged_and_kept():
    sizes = [60, 62, 64, 66, 68, 1000]
    recs = [_rec(f"u{i}", "w " * n) for i, n in enumerate(sizes)]
    kept, audit = clean_records(recs)
    assert audit["final"] == 6 and audit["flagged_outliers"] == 1
    assert [r["is_outlier"] for r in kept][-1] is True


def test_outliers_skipped_with_fewer_than_four():
    assert iqr_fences([60, 70, 5000]) is None
    kept, audit = clean_records([_rec(f"u{i}", "w " * n) for i, n in enumerate([60, 70, 5000])])
    assert audit["flagged_outliers"] == 0


def test_language_heuristic():
    assert detect_language("This is English text") == "en"
    assert detect_language("یہ اردو متن ہے") == "ur"
    assert detect_language("1234 ...") == "unknown"


def test_config_defaults():
    assert config.MIN_WORDS == 50 and config.IQR_K == 1.5


def test_merge_jsonl_appends_and_replaces(tmp_path):
    from data_scripts.utils import merge_jsonl, read_jsonl
    p = tmp_path / 'r.jsonl'
    assert merge_jsonl(p, [{'source_uri': 'a', 'v': 1}, {'source_uri': 'b', 'v': 1}]) == (2, 0, 2)
    assert merge_jsonl(p, [{'source_uri': 'b', 'v': 2}, {'source_uri': 'c', 'v': 1}]) == (1, 1, 3)
    assert [(r['source_uri'], r['v']) for r in read_jsonl(p)] == [('a', 1), ('b', 2), ('c', 1)]
