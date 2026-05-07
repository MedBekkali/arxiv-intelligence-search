"""SPECTER2 + FAISS semantic recommender.

Replaces V2's TF-IDF cosine similarity with transformer-based semantic search.

Usage:
    from arxiv_intel.recommender import Recommender

    rec = Recommender()                    # loads model + index + metadata once
    results = rec.recommend(abstract, top_k=10)
    # → DataFrame with columns:
    #   id, title, authors, year, first_cat, similarity, abs_url, pdf_url
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .links import abs_url, pdf_url


# Default artifact paths (resolved relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX = _PROJECT_ROOT / "models" / "v3_faiss.index"
_DEFAULT_META = _PROJECT_ROOT / "models" / "v3_papers_meta.parquet"
_DEFAULT_MODEL = "allenai/specter2_base"
_DEFAULT_MAX_SEQ_LENGTH = 256


class Recommender:
    """Semantic paper recommender using SPECTER2 embeddings + FAISS HNSW index."""

    def __init__(
        self,
        index_path: Path | str = _DEFAULT_INDEX,
        meta_path: Path | str = _DEFAULT_META,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load the SPECTER2 model (encodes new queries)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = _DEFAULT_MAX_SEQ_LENGTH

        # Load the prebuilt FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load the aligned metadata: row i of metadata = vector i in the index
        self.meta = pd.read_parquet(meta_path)

        # Sanity check: index size must match metadata size
        if self.index.ntotal != len(self.meta):
            raise ValueError(
                f"Index size ({self.index.ntotal}) != metadata size ({len(self.meta)}). "
                f"Index and metadata are out of sync — rebuild one of them."
            )

    def recommend(self, abstract: str, top_k: int = 10) -> pd.DataFrame:
        """Return the top_k most semantically similar papers.

        Args:
            abstract: The query text. Can be a full abstract, a title+abstract,
                      or a paraphrased query. SPECTER2 handles all of these.
            top_k: How many papers to return.

        Returns:
            DataFrame sorted by descending similarity, with columns:
            id, title, authors, year, first_cat, similarity, abs_url, pdf_url
        """
        if not abstract or not abstract.strip():
            raise ValueError("Empty query.")

        # Encode the query → 1 × 768 vector
        query_vec = self.model.encode(
            [abstract],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Normalize so inner-product search = cosine similarity
        # (the index was built with normalized vectors, so we must match)
        faiss.normalize_L2(query_vec)

        # Search the index. distances = cosine similarities (because we normalized).
        # indices = row positions in self.meta
        distances, indices = self.index.search(query_vec, k=top_k)
        distances = distances[0]   # shape (top_k,)
        indices = indices[0]

        # Build the result DataFrame from the metadata
        result = self.meta.iloc[indices].copy().reset_index(drop=True)
        result["similarity"] = distances
        result["abs_url"] = result["id"].map(abs_url)
        result["pdf_url"] = result["id"].map(pdf_url)

        return result

    def __repr__(self) -> str:
        return (
            f"Recommender(model='allenai/specter2_base', "
            f"papers={self.index.ntotal:,}, device='{self.device}')"
        )