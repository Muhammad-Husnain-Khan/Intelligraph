"""Contrast for every token pair actually used in web/src/styles.css: ``python -m scripts.contrast``."""
from __future__ import annotations

T = {"ink-900": "#151B2B", "ink-800": "#1D2438", "ink-700": "#2A3350", "ivory": "#F2EBDD", "ivory-dim": "#C9C0AE",
     "vermilion": "#E6533C", "vermilion-text": "#B8341F", "chartreuse": "#C5D85B", "brass": "#B69A69",
     "brass-text": "#7A5F2E", "slip-hover": "#E9E0CD", "heading-grey": "#4A4A4A"}
# (foreground, background, where it is used, minimum ratio)
PAIRS = [
    ("ivory", "ink-900", "body text on desk", 4.5),
    ("ivory-dim", "ink-900", "secondary text on desk", 4.5),
    ("ivory-dim", "ink-800", "status popover text", 4.5),
    ("brass", "ink-900", "small-caps labels, rules on desk", 4.5),
    ("ink-900", "ivory", "answer prose and slips", 4.5),
    ("ink-900", "slip-hover", "slip hover state", 4.5),
    ("brass-text", "ivory", "footnote provenance, numerals on paper", 4.5),
    ("brass-text", "slip-hover", "slip provenance on hover", 4.5),
    ("vermilion-text", "ivory", "graph-reached citation numeral", 4.5),
    ("heading-grey", "ivory", "slip section heading", 4.5),
    ("ink-900", "chartreuse", "highlighter marks / selected row", 4.5),
    ("chartreuse", "ink-900", "focus ring, selected node, edge (non-text, 3:1)", 3.0),
    ("vermilion", "ink-900", "bridge node, error text on desk", 3.0),
    ("ivory", "ink-900", "selected tab / radio on", 4.5),
    ("ink-900", "ivory", "segmented control: active", 4.5),
    ("ivory-dim", "ink-900", "disabled controls (opacity .55 is exempt, shown for info)", 3.0),
    ("ink-700", "ink-900", "hairlines (decorative)", 1.0),
]


def lum(h: str) -> float:
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a: str, b: str) -> float:
    la, lb = sorted((lum(T[a]), lum(T[b])), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def main() -> int:
    bad = 0
    for fg, bg, use, need in PAIRS:
        r = ratio(fg, bg)
        ok = r >= need
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {r:5.2f}:1 (min {need})  {fg} on {bg}  - {use}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
