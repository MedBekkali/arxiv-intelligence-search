# arXiv Intelligence Search

**Semantic search and grounded Q&A over 902,645 arXiv Computer Science papers.**

This is an ML/NLP portfolio project built as a working Streamlit product: classify a paper abstract, find semantically similar research with SPECTER2 + FAISS, and ask citation-grounded questions answered with Anthropic Claude.

## Why It Matters

Recruiters can read this as: **a full-stack ML search system over nearly one million scientific abstracts, with classical ML, transformer embeddings, vector search, RAG, and LoRA fine-tuning work in one coherent project.**

Technical reviewers can inspect the pipeline end to end: raw arXiv metadata is cleaned into CS-only records, classical baselines are trained and evaluated, SPECTER2 embeddings are generated, a FAISS HNSW index powers semantic retrieval, and RAG answers are grounded in retrieved papers.

ML engineers can audit the methodology: stratified splits, train-only vectorizer fitting, multi-label stratification for LoRA data prep, train-only class imbalance weights, threshold sweeps, per-class metrics, and explicit artifact boundaries.

## Features

| Area | Implementation | Status |
| --- | --- | --- |
| Streamlit app | Three-tab interface for classification, semantic search, and RAG Q&A | Implemented |
| V1 classifier | TF-IDF bigrams + Logistic Regression over 39 CS categories | Implemented |
| Semantic embeddings | SPECTER2 vectors for title + abstract text | Implemented |
| Vector search | FAISS HNSW index with cosine similarity via normalized inner product | Implemented |
| RAG Q&A | SPECTER2 retrieval + Claude generation with numbered paper citations | Implemented |
| LoRA fine-tuning | SPECTER2 multi-label classifier scripts over 39 CS categories | Implemented; final test metrics TBD |

## Architecture Overview

```text
arXiv metadata
    -> CS-only cleaning and deduplication
    -> 902,645 title/abstract records
        -> V1 TF-IDF + Logistic Regression classifier
        -> SPECTER2 embedding generation
            -> FAISS HNSW semantic index
                -> similar-paper search
                -> RAG context retrieval
                    -> Anthropic Claude answer with citations
        -> multi-label train/val/test splits
            -> SPECTER2 + LoRA classifier training
```

For a fuller technical walkthrough, see [docs/architecture.md](docs/architecture.md).

## Metrics

| Component | Dataset / Split | Metric | Result |
| --- | --- | --- | --- |
| V1 classifier | 80/20 stratified split, primary CS category | Accuracy | 0.7345 |
| V1 classifier | Same held-out test set | Macro F1 | 0.6068 |
| V1 classifier | Same held-out test set | Weighted F1 | 0.7289 |
| V1 classifier | Training data | Train records | 722,116 |
| V1 classifier | Test data | Test records | 180,529 |
| LoRA classifier | Validation threshold sweep, Experiment A | Macro F1 | 0.5709 |
| LoRA classifier | Final selected test evaluation | Macro / micro / weighted F1 | TBD |
| Retrieval / RAG | Qualitative inspection | Grounded citations | Implemented; benchmark TBD |

The V1 metrics are saved in `models/v1_metrics.json`. Final LoRA test metrics are intentionally marked TBD because they are not present as finalized public results in this repository.

## Screenshots

Screenshots are intentionally left as placeholders until the app is run with local artifacts:

- Classifier tab: `docs/screenshots/classify.png` TBD
- Semantic search tab: `docs/screenshots/search.png` TBD
- RAG Q&A tab: `docs/screenshots/rag.png` TBD

## Repository Map

```text
.
├── app.py                         # Streamlit app entry point
├── src/arxiv_intel/               # Recommender, RAG, arXiv link helpers
├── scripts/                       # Data, embedding, FAISS, RAG, LoRA workflows
├── notebooks/                     # Data prep, EDA, V1 baseline notebook work
├── docs/                          # Architecture, methodology, artifact notes
├── models/                        # Small tracked metadata + ignored generated artifacts
├── data/                          # Ignored local datasets
└── README.md
```

## Artifact Note

The full dataset, embedding matrix, FAISS index, and trained model artifacts are **not stored on GitHub** because they are large generated files. Examples:

- Raw arXiv metadata: about 5 GB
- Cleaned CS parquet: about 655 MB
- SPECTER2 embeddings: about 2.6 GB
- FAISS HNSW index: about 2.9 GB

Small metadata such as `models/v1_metrics.json` and `models/v3_categories.json` is tracked. See [docs/artifacts.md](docs/artifacts.md) for the artifact inventory and regeneration map.

## Quickstart

Clone and enter the project:

```bash
git clone <repo-url>
cd arxiv-intelligence-search
```

Create the local environment:

```bash
uv sync
```

If your environment does not already include the V3/RAG/LoRA dependencies, install the runtime extras used by the scripts:

```bash
uv pip install faiss-cpu anthropic python-dotenv transformers peft iterative-stratification pyarrow tqdm
```

Configure Anthropic credentials for RAG:

```bash
cp .env.example .env
```

Then edit `.env` locally:

```text
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-haiku-4-5
```

Run the Streamlit app after the required local artifacts have been generated or restored:

```bash
streamlit run app.py
```

Core regeneration order:

```bash
python scripts/prepare_multilabel_data.py
python scripts/embed_papers.py
python scripts/build_faiss_index.py
python scripts/train_lora_classifier.py --dry-run
```

The full embedding and FAISS steps require substantial disk space, RAM, and time. GPU acceleration is strongly recommended for embedding generation and LoRA training.

## Documentation

- [docs/architecture.md](docs/architecture.md): system pipeline and module-level architecture
- [docs/methodology.md](docs/methodology.md): dataset, splits, metrics, evaluation discipline
- [docs/artifacts.md](docs/artifacts.md): generated files, sizes, tracking status, regeneration source
- [docs/embed_papers_walkthrough.md](docs/embed_papers_walkthrough.md): detailed SPECTER2 embedding walkthrough
- [docs/prepare_multilabel_data_walkthrough.md](docs/prepare_multilabel_data_walkthrough.md): detailed multi-label data prep walkthrough

## Limitations

- Full generated artifacts are excluded from GitHub, so a fresh clone cannot run every feature until artifacts are regenerated or restored.
- RAG requires an Anthropic API key and incurs API usage costs.
- Retrieval quality is currently validated qualitatively; a formal retrieval benchmark is TBD.
- Final LoRA test-set metrics are TBD and should not be inferred from validation notes.
- The V1 classifier predicts only the primary CS category; the LoRA workflow addresses multi-label classification but is documented separately.
- Running full SPECTER2 embedding generation over 902,645 papers is hardware-intensive.

## Portfolio Positioning

This project is designed for French AI/ML alternance applications. It demonstrates:

- Applied NLP and information retrieval at realistic corpus scale
- Classical ML baseline design and evaluation
- Transformer embedding pipelines with checkpointing and reproducibility concerns
- Vector search with FAISS HNSW
- Retrieval-augmented generation with grounded citations
- Multi-label classification setup for scientific-paper categories
- Practical engineering around large artifacts, local secrets, and deployment constraints

Author: Mohammed BEKKALI, ESAIP Angers, AI specialization.
