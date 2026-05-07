# Methodology

This document summarizes the ML methodology behind arXiv Intelligence Search. It is written for reviewers who want to inspect whether the results and workflows are credible.

## Dataset

The project uses arXiv metadata filtered to Computer Science papers.

Observed cleaned corpus:

- Records: 902,645
- Modalities: title, abstract, authors, year, arXiv ID, category metadata
- Primary label: `first_cat`
- Multi-label source: `cs_cats`
- Label space: 39 canonical CS categories

The cleaned corpus is stored locally as `data/processed/arxiv_cs_clean.parquet` and is not tracked because it is about 655 MB.

## Cleaning

The data preparation workflow removes or checks for:

- Missing abstracts or category fields
- Duplicate abstracts that could create train/test leakage
- Abstract length outliers
- Non-CS category handling for the CS-only label space

The main cleaned output is the source for the V1 classifier, SPECTER2 embedding generation, FAISS metadata, and multi-label LoRA preparation.

## V1 Single-Label Split

The V1 classifier predicts the primary CS category, `first_cat`.

Split discipline:

- 80/20 train/test split
- Stratified by encoded primary category
- Fixed random seed
- TF-IDF vectorizer fit on training text only
- Test set transformed with the training vocabulary

This prevents vocabulary leakage from test data into the feature extractor.

V1 saved metrics:

| Metric | Value |
| --- | ---: |
| Test accuracy | 0.7345 |
| Test macro F1 | 0.6068 |
| Test weighted F1 | 0.7289 |
| Train records | 722,116 |
| Test records | 180,529 |

## Semantic Retrieval

The retrieval system uses SPECTER2 embeddings rather than sparse keyword vectors.

Embedding choices:

- Input format: `title [SEP] abstract`
- Encoder: `allenai/specter2_base`
- Vector size: 768 dimensions
- Storage dtype: float32
- Query-time model: same SPECTER2 encoder

Retrieval choices:

- Vectors are L2-normalized
- FAISS inner product is used as cosine similarity
- HNSW index is used for approximate nearest-neighbor search
- Returned metadata includes paper title, arXiv ID, category, year, authors, abstract/PDF links, and similarity score

Current retrieval evaluation is primarily qualitative. A formal benchmark is TBD.

## RAG Discipline

The RAG pipeline retrieves papers before generation and asks the language model to answer only from the provided excerpts.

RAG guardrails:

- Retrieved papers are numbered
- The answer must cite sources using `[#1]`, `[#2]`, etc.
- The prompt tells the model not to invent facts outside the excerpts
- If context is insufficient, the model should say so plainly
- Token usage and estimated cost are returned

RAG quality still depends on retrieval quality, source coverage, and the LLM following the prompt. It should be treated as a grounded assistant, not as a verified scientific authority.

## Multi-Label Strategy

V1 predicts one primary category, but real arXiv papers can be cross-listed across multiple CS categories. The LoRA workflow treats classification as multi-label.

Transformation:

```text
cs_cats = ["cs.CV", "cs.LG"]
    -> label_indices = [index("cs.CV"), index("cs.LG")]
    -> multi-hot target during training
```

The repository stores sparse `label_indices` in parquet files because dense 39-position vectors would mostly contain zeros. The training collator expands sparse indices into multi-hot tensors batch by batch.

## Multi-Label Splits

The LoRA data preparation script creates train/validation/test splits with multi-label stratification.

Default split:

| Split | Approx records | Role |
| --- | ---: | --- |
| Train | 721,899 | Model fitting |
| Validation | 90,406 | Threshold selection and model selection |
| Test | 90,340 | Final evaluation only |

The split is performed in two stages:

```text
full dataset
    -> trainval + test
    -> train + validation from trainval
```

The code asserts that train, validation, and test indices do not overlap and together cover the full filtered dataset.

## Train-Only `pos_weight`

The LoRA classifier uses `BCEWithLogitsLoss` for multi-label classification. Because category frequencies are imbalanced, the data-prep script computes a per-class `pos_weight`.

Important discipline:

- `pos_weight` is computed from training labels only
- Validation and test label frequencies are not used for training loss construction
- This avoids leaking evaluation-set statistics into the training process

Some later experiments clamp very large `pos_weight` values to reduce rare-class gradient domination and improve probability calibration. Those are experiment choices and should be reported alongside results.

## Metrics

The project uses different metrics for different tasks:

| Task | Metrics |
| --- | --- |
| V1 primary-category classifier | Accuracy, macro F1, weighted F1, per-class report |
| LoRA multi-label classifier | Macro F1, micro F1, weighted F1, precision, recall, empty-prediction rate, average predicted labels |
| Retrieval | Qualitative inspection currently; formal retrieval metrics TBD |
| RAG | Citation grounding and source inspection currently; automated factuality benchmark TBD |

Macro F1 is especially important for imbalanced categories because it gives each class equal weight. Weighted F1 is useful for understanding aggregate performance over the natural category distribution.

## Threshold Selection

The LoRA classifier outputs independent sigmoid probabilities for each class. A global threshold converts probabilities into label predictions.

Validation threshold sweep:

- Multiple candidate thresholds are evaluated
- Best thresholds are selected by macro, micro, and weighted F1
- Threshold sweep CSV files are saved for inspection
- Test data should be used only after model and threshold choices are finalized

Final LoRA test metrics are TBD in this repository.

## Evaluation Discipline

The intended discipline is:

1. Use train data to fit model parameters.
2. Use validation data for threshold selection, early stopping, and model selection.
3. Use test data once for final reporting.
4. Keep generated artifacts and run logs traceable.
5. Mark incomplete final results as TBD rather than promoting validation numbers as test performance.

That is why the README reports V1 test metrics directly, but marks final LoRA metrics as TBD.
