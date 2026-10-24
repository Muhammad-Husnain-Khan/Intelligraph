"""Offline tests for the web collector: SSRF guard, HTML extraction, manifest reading."""
import pytest

from data_scripts.collect_web import extract_text, read_manifest
from data_scripts.make_report_numbers import tex_escape
from data_scripts.utils import is_safe_url, source_id


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.5/admin",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "http://224.0.0.1/",
        "ftp://example.com/file",
        "file:///etc/passwd",
        "http:///nohost",
    ],
)
def test_ssrf_rejects_unsafe(url):
    ok, _ = is_safe_url(url)
    assert not ok


def test_ssrf_accepts_public_ip_literal():
    ok, _ = is_safe_url("https://8.8.8.8/")
    assert ok


HTML = """
<html><head><title>Fallback</title><script>var x=1;</script></head><body>
<nav>Menu stuff</nav>
<h1 id="firstHeading">Machine learning</h1>
<div id="mw-content-text">
  <h2>History<span class="mw-editsection">[edit]</span></h2>
  <p>ML is a field.<sup class="reference">[1]</sup> It learns.</p>
  <ul><li>Item one</li><li>Item two</li></ul>
  <div class="navbox"><p>Navbox text</p></div>
</div>
<footer>Footer text</footer></body></html>
"""


def test_extract_text_cleans_and_keeps_paragraphs():
    title, text = extract_text(HTML)
    assert title == "Machine learning"
    assert text.split("\n\n") == ["History", "ML is a field. It learns.", "Item one", "Item two"]
    for junk in ("Menu", "Footer", "Navbox", "[edit]", "[1]", "var x"):
        assert junk not in text


def test_read_manifest_skips_comments_and_blanks(tmp_path):
    f = tmp_path / "urls.csv"
    f.write_text(
        "url,expected_language,notes\n\n# comment\nhttps://a.example/x,en,n\n", encoding="utf-8"
    )
    rows = read_manifest(f)
    assert rows == [{"url": "https://a.example/x", "expected_language": "en", "notes": "n"}]


def test_source_id_and_tex_escape():
    assert source_id("abc") == "a9993e364706"
    assert tex_escape("a_b & 50%") == r"a\_b \& 50\%"
