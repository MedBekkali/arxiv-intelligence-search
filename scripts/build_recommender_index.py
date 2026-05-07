"""One-time builder for the V2 recommender index.

Usage:
    python scripts/build_recommender_index.py

Reads:
    data/processed/papers_clean.parquet   (the V1 clean corpus)
    models/tfidf_vectorizer.pkl           (the V1 fitted vectorizer)

Writes:
    models/v2_recommender_matrix.npz      (L2-normalised TF-IDF matrix)
    models/v2_recommender_meta.parquet    (id, title, authors, year, first_cat)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA = Path("data/processed/arxiv_cs_clean.parquet")
VEC = Path("models/v1_tfidf_vectorizer.pkl")
OUT_MAT = Path("models/v2_recommender_matrix.npz")
OUT_META= Path("models/v2_recommender_meta.parquet")

META_COLS = ["id", "title", "authors", "year", "first_cat"]


def main() -> None:
    print("Loading corpus …")
    df = pd.read_parquet(DATA)

    print("Loading vectorizer …")
    with open(VEC, "rb") as f:
        vectorizer = pickle.load(f)

    print(f"Vectorising {len(df):,} abstracts …")
    matrix = vectorizer.transform(df["abstract"].fillna(""))

    print("L2-normalising …")
    norms = np.asarray(matrix.power(2).sum(axis=1)).ravel() ** 0.5
    norms[norms == 0] = 1          # avoid division by zero
    matrix = matrix.multiply(1.0 / norms[:, np.newaxis]).tocsr()

    print(f"Saving matrix → {OUT_MAT}")
    OUT_MAT.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(OUT_MAT, matrix)

    print(f"Saving metadata → {OUT_META}")
    keep = [c for c in META_COLS if c in df.columns]
    df[keep].to_parquet(OUT_META, index=False)

    print("Done.")


if __name__ == "__main__":
    main()