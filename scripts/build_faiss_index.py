from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EMBEDDINGS_PATH = PROJECT_ROOT / 'models' / 'v3_specter2_embeddings.npy'
INDEX_PATH = PROJECT_ROOT / 'models' / 'v3_faiss.index'

# HNSW parameters
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

def main() -> None:
    print(f"Loading embeddings from {EMBEDDINGS_PATH.name}...")
    t0 = time.time()
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded shape {embeddings.shape} in {time.time()-t0:.1f}s")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    print("L2-normalizing vectors (so inner product = cosine similarity)...")
    faiss.normalize_L2(embeddings)

    n_vectors, dim = embeddings.shape
    print(f"Building HNSW index ({n_vectors:,} vectors, dim={dim}, M={HNSW_M})...")
    t0 = time.time()
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(embeddings)
    build_time = time.time() - t0
    print(f"Index built in {build_time/60:.1f} min ({n_vectors / build_time:.0f} vectors/second)")

    # Save
    print(f"Saving index to {INDEX_PATH.name}...")
    faiss.write_index(index, str(INDEX_PATH))
    size_mb = INDEX_PATH.stat().st_size / 1e6
    print(f"Saved {size_mb:.1f} MB")

    # Quick sanity test: query for the first vector's nearest neighbors
    print("\nSanity test: nearest neighbors of paper #0")
    distances, indices = index.search(embeddings[:1], k=5)
    print(f"  Top 5 neighbor indices: {indices[0]}")
    print(f"  Similarities: {distances[0]}")
    print(f"  (paper 0 should be its own #1 neighbor with similarity ~1.0)")


if __name__ == '__main__':
    main()
