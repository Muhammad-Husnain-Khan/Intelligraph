from data_scripts.chunk_data import chunk_record, is_heading, window_text


def _sent(tag: str, n_words: int) -> str:
    return " ".join(f"{tag}{i}" for i in range(n_words - 1)) + " end."


def _rec(text: str, source_type: str = "website") -> dict:
    return {"source_id": "abc123", "source_uri": "u", "title": "T",
            "source_type": source_type, "language": "en", "text": text}


def test_heading_heuristic():
    assert is_heading("History")
    assert not is_heading("This is a normal sentence.")
    assert not is_heading("word " * 20)


def test_sections_follow_headings_and_drop_references():
    body = " ".join(["word"] * 80) + "."
    text = f"Intro\n\n{body}\n\nHistory\n\n{body}\n\nReferences\n\n{body}"
    chunks = chunk_record(_rec(text), 300, 50, 50)
    assert [c["section_heading"] for c in chunks] == ["Intro", "History"]
    assert chunks[0]["chunk_id"] == "abc123-0000" and chunks[1]["chunk_index"] == 1


def test_long_section_windowed_with_overlap_and_cap():
    text = " ".join(_sent(f"s{k}_", 20) for k in range(40))  # 800 words
    windows = window_text(text, 300, 50, 50)
    assert len(windows) >= 3
    assert all(len(w.split()) <= 300 + 20 + 50 for w in windows)
    # overlap: last words of window i appear at the start of window i+1
    assert windows[0][-40:] in windows[1]  # tail of window 0 is carried into window 1
    assert "s39_0" in windows[-1]  # nothing lost at the end


def test_small_tail_is_not_a_separate_chunk():
    text = " ".join(_sent(f"a{k}_", 20) for k in range(16))  # 320 words
    windows = window_text(text, 300, 50, 50)
    assert all(len(w.split()) >= 50 for w in windows)


def test_pdf_has_no_heading_detection():
    text = "Abstract\n\n" + " ".join(["word"] * 100) + "."
    chunks = chunk_record(_rec(text, "pdf"), 300, 50, 50)
    assert len(chunks) == 1 and chunks[0]["section_heading"] == ""
    assert "Abstract" in chunks[0]["text"]


def test_huge_unpunctuated_text_is_hard_split():
    chunks = chunk_record(_rec(" ".join(["w"] * 1000), "pdf"), 300, 50, 50)
    assert len(chunks) >= 3 and all(c["word_count"] <= 400 for c in chunks)
