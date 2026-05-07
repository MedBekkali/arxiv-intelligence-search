# Architecture

arXiv Intelligence Search is organized as a local ML product: offline data and model pipelines generate artifacts, then a Streamlit app loads those artifacts for interactive classification, semantic search, and RAG Q&A.

## Pipeline Overview

```text
arXiv metadata snapshot
    -> CS paper extraction
    -> cleaning, deduplication, length filtering
    -> 902,645 cleaned CS records
        -> V1 single-label classifier
        -> SPECTER2 embedding pipeline
            -> FAISS HNSW index
                -> semantic recommendations
                -> RAG retrieval context
                    -> Anthropic Claude answer
        -> multi-label split preparation
            -> SPECTER2 + LoRA training workflow
```

The app entry point is `app.py`. The reusable runtime modules live in `src/arxiv_intel/`.

## V1 Classifier

Purpose: classify an abstract into one primary arXiv CS category.

Flow:

```text
abstract text
    -> TF-IDF vectorizer
    -> Logistic Regression classifier
    -> top-k category probabilities
```

Key properties:

- Label space: 39 CS categories from `first_cat`
- Input: abstract text
- Features: TF-IDF, unigrams + bigrams, 50k max features
- Model: Logistic Regression
- Split: 80/20 stratified train/test split
- Saved artifacts: vectorizer, classifier, label encoder, metrics JSON

The V1 model is intentionally classical. It provides a fast, inspectable baseline before the transformer retrieval and LoRA workflows.

## SPECTER2 Embeddings

Purpose: represent each paper as a dense semantic vector suitable for similarity search.

Flow:

```text
title + [SEP] + abstract
    -> SPECTER2 encoder
    -> 768-dimensional float32 embedding
    -> aligned metadata file
```

Key properties:

- Model: `allenai/specter2_base`
- Input text: paper title and abstract
- Max sequence length: 256 tokens by default
- Output shape: about `(902645, 768)`
- Storage: `models/v3_specter2_embeddings.npy`
- Metadata: `models/v3_papers_meta.parquet`

The embedding script is built for long-running jobs: it supports chunking, checkpointing, resume behavior, GPU detection, and GPU temperature logging.

## FAISS Retrieval

Purpose: search the paper corpus by semantic meaning rather than keyword overlap.

Flow:

```text
query text
    -> SPECTER2 query embedding
    -> L2 normalization
    -> FAISS HNSW inner-product search
    -> top-k paper metadata + arXiv links
```

Key properties:

- Index type: FAISS `IndexHNSWFlat`
- Similarity: cosine similarity implemented as inner product over normalized vectors
- Default index file: `models/v3_faiss.index`
- Runtime wrapper: `src/arxiv_intel/recommender.py`

The `Recommender` class loads the SPECTER2 model, the FAISS index, and aligned metadata once, then serves repeated search queries.

## RAG Module

Purpose: answer natural-language questions using only retrieved arXiv paper excerpts.

Flow:

```text
user question
    -> Recommender top-k retrieval
    -> numbered context block
    -> Anthropic Claude message
    -> answer with [#1], [#2] style citations
```

Key properties:

- Runtime wrapper: `src/arxiv_intel/rag.py`
- Retrieval: reuses the same SPECTER2 + FAISS stack
- LLM provider: Anthropic Claude
- Credentials: loaded from local `.env`
- Output: answer text, source metadata, token usage, estimated cost

The system prompt requires grounded answers and asks the model to avoid unsupported claims when retrieved excerpts are insufficient.

## LoRA Classifier

Purpose: fine-tune a SPECTER2-based classifier for multi-label CS category prediction.

Flow:

```text
cleaned corpus
    -> canonical 39-category label space
    -> cs_cats converted to label_indices
    -> multi-label stratified train/val/test split
    -> train-only pos_weight vector
    -> SPECTER2 encoder + LoRA adapters + classifier head
    -> validation threshold sweeps
    -> best checkpoint directories
```

Key properties:

- Label type: multi-label, because arXiv papers can have multiple CS categories
- Loss: `BCEWithLogitsLoss`
- Class imbalance: `pos_weight`, computed on train only
- Tuning: validation threshold sweeps
- Saved outputs: adapters, classifier head, tokenizer, categories, training config, metric histories

Final public LoRA test metrics are TBD in the current repository. Validation experiment notes exist, but final test performance should not be inferred from them.

## Runtime Feature Dependencies

| Feature | Required local artifacts |
| --- | --- |
| V1 classification | `models/v1_tfidf_vectorizer.pkl`, `models/v1_logreg.pkl`, `models/v1_label_encoder.pkl` |
| Semantic search | `models/v3_faiss.index`, `models/v3_papers_meta.parquet`, SPECTER2 model availability |
| RAG Q&A | Semantic search artifacts, `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` |
| LoRA evaluation | LoRA checkpoint directory, `data/processed/val.parquet` or `test.parquet` |

Because the largest artifacts are ignored, a fresh clone shows architecture and methodology immediately but requires artifact regeneration or restoration for full runtime behavior.
