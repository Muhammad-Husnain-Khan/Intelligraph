"""Generate comprehensive EDA figures from cleaned records and chunks.

Visualizations:
1. Box / Violin Plot: Word Count by Source Type
2. Log-Scale Word-Count Histogram (with 50/300-word chunk thresholds & outliers)
3. Top 20 Terms Bar Chart (domain keywords, stopwords removed)
4. 2D Projection Scatter Plot (TF-IDF with PCA or TruncatedSVD)
5. Scatter Plot: Word Count vs. Average Sentence Length
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .clean_data import iqr_fences
from .utils import PipelineError, get_logger, read_jsonl

logger = get_logger(__name__)

# Modern, harmonious palette
PALETTE = {
    "website": "#2b5c8f",
    "pdf": "#d95f02",
    "docx": "#2ca02c",
    "txt": "#7570b3",
    "primary": "#2b5c8f",
    "accent": "#e74c3c",
    "neutral": "#555555",
    "grid": "#eaeaea",
}

_STOPWORDS = {
    # Common English stopwords
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "using", "used", "use", "one", "two", "may", "can", "well",
    "et", "al", "fig", "figure", "table", "section", "page", "per", "rs", "app", "click", "dial",
}


def _save(fig, path: Path) -> None:
    """Save a figure as PNG with retry on Windows file lock."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    for attempt in range(1, 6):
        try:
            path.write_bytes(buf.getvalue())
            return
        except OSError:
            if attempt == 5:
                raise
            time.sleep(0.5 * attempt)


def plot_word_count_by_source_type(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 1: Box / Violin Plot of word counts across source types."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    types = [t for t in sorted(df["source_type"].unique())]
    data_by_type = [df[df["source_type"] == t]["word_count"].values for t in types]

    # Use violin plot with data points overlaid
    try:
        parts = ax.violinplot(
            data_by_type,
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE["primary"])
            pc.set_edgecolor("#1c3b5e")
            pc.set_alpha(0.6)
        if "cmeans" in parts:
            parts["cmeans"].set_color(PALETTE["accent"])
        if "cmedians" in parts:
            parts["cmedians"].set_color("#27ae60")
    except Exception:
        # Fallback to boxplot if violin fails on small samples
        ax.boxplot(data_by_type, patch_artist=True, labels=types)

    # Add jittered points for clarity
    for i, values in enumerate(data_by_type, start=1):
        jitter = np.random.normal(0, 0.04, size=len(values)) if len(values) > 1 else np.array([0])
        ax.scatter(i + jitter, values, color="#1c3b5e", alpha=0.8, s=40, zorder=3)

    ax.set_xticks(range(1, len(types) + 1))
    ax.set_xticklabels([f"{t.upper()} (n={len(df[df['source_type'] == t])})" for t in types], fontweight="bold")
    ax.set_title("Word Count Distribution by Source Type", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Source Type", fontsize=10, fontweight="bold")
    ax.set_ylabel("Word Count (Log Scale)", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    _save(fig, out_path)
    plt.close(fig)


def plot_log_word_count_histogram(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 2: Log-scale word count histogram with chunk thresholds & outliers."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    counts = df["word_count"].values

    bins = np.geomspace(max(1, counts.min()), max(10, counts.max()), num=min(20, max(len(df), 4)))
    ax.hist(counts, bins=bins, color=PALETTE["primary"], edgecolor="white", alpha=0.85, label="Documents")
    ax.set_xscale("log")

    # Mark chunking thresholds (50 words min, 300 words max cap)
    ax.axvline(50, color="#f39c12", linestyle="--", linewidth=1.8, label="Min Chunk Size (50 words)")
    ax.axvline(300, color="#27ae60", linestyle="--", linewidth=1.8, label="Chunk Cap (300 words)")

    # Outliers
    outliers = df[df.get("is_outlier", False)]
    if not outliers.empty:
        ax.scatter(
            outliers["word_count"],
            [0.2] * len(outliers),
            color=PALETTE["accent"],
            marker="v",
            s=70,
            zorder=4,
            label=f"IQR Outliers ({len(outliers)})",
        )

    fences = iqr_fences(counts.tolist())
    if fences and fences[1] > 0:
        ax.axvline(fences[1], color=PALETTE["accent"], linestyle=":", linewidth=1.5, label=f"Upper Fence ({int(fences[1])}w)")

    ax.set_title("Log-Scale Document Word-Count Distribution", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Words per Document (Log Scale)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Documents", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    _save(fig, out_path)
    plt.close(fig)


def plot_top_terms_barchart(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 3: Top 20 terms bar chart with stopwords removed."""
    from collections import Counter

    tokens: list[str] = []
    for text in df["text"]:
        # Word tokenizer extracting alphabetic / Urdu tokens of length >= 3
        words = re.findall(r"\b[A-Za-z\u0600-\u06FF]{3,}\b", text.lower())
        for w in words:
            if w not in _STOPWORDS and not w.isdigit():
                tokens.append(w)

    counter = Counter(tokens)
    top20 = counter.most_common(20)

    fig, ax = plt.subplots(figsize=(8, 5))
    if top20:
        terms, freqs = zip(*reversed(top20))
        bars = ax.barh(terms, freqs, color=PALETTE["primary"], edgecolor="none", height=0.7)
        ax.bar_label(bars, padding=4, fontsize=9, color="#222222")
        ax.set_xlim(0, max(freqs) * 1.15)
    else:
        ax.text(0.5, 0.5, "Insufficient term data", ha="center", va="center")

    ax.set_title("Top 20 Domain Terms Across Corpus (Stopwords Removed)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Term Frequency", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    _save(fig, out_path)
    plt.close(fig)


def plot_2d_projection(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 4: 2D TF-IDF Projection Scatter Plot (PCA / SVD) colored by source type."""
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # If small number of documents, fall back to chunk level for rich clusters
    source_texts = df["text"].tolist()
    labels = df["source_type"].tolist()

    if len(source_texts) < 4:
        # Augment with chunks if available for a meaningful 2D map
        chunks_file = config.PROCESSED_DIR / "chunks.jsonl"
        if chunks_file.exists():
            chunks = read_jsonl(chunks_file)
            if chunks:
                source_texts = [c["text"] for c in chunks]
                labels = [c["source_type"] for c in chunks]

    n_samples = len(source_texts)
    if n_samples >= 2:
        try:
            vec = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
            tfidf = vec.fit_transform(source_texts).toarray()
            
            n_components = min(2, tfidf.shape[0], tfidf.shape[1])
            if n_components >= 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(tfidf)
            else:
                coords = np.column_stack([np.arange(n_samples), np.zeros(n_samples)])

            unique_labels = sorted(set(labels))
            for ul in unique_labels:
                idx = [i for i, l in enumerate(labels) if l == ul]
                color = PALETTE.get(ul, "#888888")
                ax.scatter(
                    coords[idx, 0],
                    coords[idx, 1],
                    label=f"{ul.upper()} ({len(idx)})",
                    color=color,
                    s=60,
                    alpha=0.85,
                    edgecolors="white",
                    linewidth=0.5,
                )
        except Exception as exc:
            logger.warning("TF-IDF 2D projection error: %s", exc)
            ax.text(0.5, 0.5, "Unable to project 2D clusters", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Requires >= 2 documents for 2D projection", ha="center", va="center")

    ax.set_title("2D TF-IDF Semantic Projection (PCA Clustering)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Component 1 (Principal Dimension)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Component 2 (Orthogonal Dimension)", fontsize=10, fontweight="bold")
    ax.legend(title="Source Type", loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    _save(fig, out_path)
    plt.close(fig)


def plot_word_count_vs_sentence_length(df: pd.DataFrame, out_path: Path) -> None:
    """Plot 5: Scatter Plot: Word Count vs. Average Sentence Length."""
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Calculate average sentence length (words per sentence)
    avg_sent_len = []
    for _, row in df.iterrows():
        s_count = max(row.get("sentence_count", 1), 1)
        w_count = row.get("word_count", 0)
        avg_sent_len.append(w_count / s_count)

    df_plot = df.copy()
    df_plot["avg_sent_len"] = avg_sent_len

    for st in sorted(df_plot["source_type"].unique()):
        sub = df_plot[df_plot["source_type"] == st]
        color = PALETTE.get(st, "#333333")
        ax.scatter(
            sub["word_count"],
            sub["avg_sent_len"],
            label=f"{st.upper()} ({len(sub)})",
            color=color,
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_title("Document Size vs. Syntactic Complexity (Sentence Length)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Total Word Count (Log Scale)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Sentence Length (Words / Sentence)", fontsize=10, fontweight="bold")
    ax.legend(title="Source Type", loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    _save(fig, out_path)
    plt.close(fig)


def run() -> None:
    """Generate the 5 new visual distribution plots into report/figures/."""
    records = read_jsonl(config.CLEAN_RECORDS)
    if not records:
        raise PipelineError(
            f"No cleaned records at {config.CLEAN_RECORDS}. Run collection and cleaning first."
        )

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)

    # 1. Box / Violin Plot: Word Count by Source Type
    plot_word_count_by_source_type(df, config.FIGURES_DIR / "word_count_by_source_type.png")

    # 2. Log-Scale Word-Count Histogram
    plot_log_word_count_histogram(df, config.FIGURES_DIR / "log_word_count_histogram.png")

    # 3. Top 20 Terms Bar Chart (Stopwords Removed)
    plot_top_terms_barchart(df, config.FIGURES_DIR / "top_20_terms.png")

    # 4. 2D Projection Scatter Plot (TF-IDF with PCA)
    plot_2d_projection(df, config.FIGURES_DIR / "tfidf_2d_projection.png")

    # 5. Scatter Plot: Word Count vs. Average Sentence Length
    plot_word_count_vs_sentence_length(df, config.FIGURES_DIR / "word_count_vs_sentence_length.png")

    logger.info("All 5 enhanced figures written to %s", config.FIGURES_DIR)


if __name__ == "__main__":
    run()
