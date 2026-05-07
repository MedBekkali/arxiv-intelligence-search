# Multi-Label Dataset Preparation — Complete Walkthrough

A reference document for `scripts/prepare_multilabel_data.py`. Read top-to-bottom for understanding. Use as study material before whiteboarding or explaining Day 6 of the project.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Architecture Diagrams](#2-architecture-diagrams)
3. [Multi-Label Classification — Deep Dive](#3-multi-label-classification--deep-dive)
4. [Label Indices vs Multi-Hot Vectors](#4-label-indices-vs-multi-hot-vectors)
5. [Why Multi-Label Stratified Split Matters](#5-why-multi-label-stratified-split-matters)
6. [Class Imbalance and `pos_weight`](#6-class-imbalance-and-pos_weight)
7. [The Leakage Question — Why Train Only](#7-the-leakage-question--why-train-only)
8. [The Script — Section by Section](#8-the-script--section-by-section)
9. [Why Specific Engineering Choices](#9-why-specific-engineering-choices)
10. [What to Whiteboard](#10-what-to-whiteboard)
11. [What Makes This Script Professional](#11-what-makes-this-script-professional)
12. [Quick Reference Card](#12-quick-reference-card)

---

## 1. The Big Picture

**What we're trying to do:** prepare the dataset for fine-tuning a SPECTER2-based classifier that can predict **multiple CS categories per paper**.

In V1, the classifier predicted only one category:

```text
paper → one label
```

Example:

```text
"Vision Transformers for Image Classification" → cs.CV
```

But arXiv papers can belong to multiple CS categories at once:

```text
paper → multiple labels
```

Example:

```text
"Vision Transformers for Image Classification" → cs.CV + cs.LG
```

So Day 6 prepares the data for a **multi-label classification** setup.

The script takes the full cleaned corpus:

```text
data/processed/arxiv_cs_clean.parquet
902,645 papers
```

and creates:

```text
data/processed/train.parquet
721,899 papers

data/processed/val.parquet
90,406 papers

data/processed/test.parquet
90,340 papers

models/v3_categories.json
39 canonical CS categories

models/v3_pos_weights.npy
39 class imbalance weights
```

The most important transformation is this:

```text
cs_cats = ["cs.CV", "cs.LG"]
```

becomes:

```text
label_indices = [11, 25]
```

where `11` and `25` are positions in the sorted category list.

Why do this? Because the training script later needs to turn those indices into a 39-dimensional multi-hot vector:

```text
[0, 0, 0, ..., 1 at cs.CV, ..., 1 at cs.LG, ..., 0]
```

That is what `BCEWithLogitsLoss` expects for multi-label classification.

**One-sentence summary:**

> This script turns the raw arXiv CS corpus into clean train/val/test files for multi-label fine-tuning, while preserving rare class proportions and computing class imbalance weights without leaking validation/test information.

---

## 2. Architecture Diagrams

### Diagram 1: Data flow from full corpus to training splits

```text
        ┌────────────────────────────────────────────────────┐
        │ data/processed/arxiv_cs_clean.parquet              │
        │ 902,645 papers                                     │
        │ columns: id, title, abstract, cs_cats, first_cat... │
        └──────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────┐
        │ Build canonical category list from first_cat        │
        │ sorted 39 CS categories                            │
        │ example: cs.AI=0, cs.AR=1, ..., cs.SY=38            │
        └──────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────┐
        │ Convert cs_cats → label_indices                    │
        │ ["cs.CV", "cs.LG"] → [11, 25]                     │
        │ drop non-CS cross-list tags like stat.ML            │
        └──────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────┐
        │ Build temporary dense multi-hot matrix Y            │
        │ shape: (902,645, 39)                                │
        │ used only for stratified splitting                  │
        └──────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────┐
        │ Multi-label stratified split                        │
        │ preserve rare/common category proportions           │
        └──────────────────────────┬─────────────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
        ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
        │ train.parquet  │ │ val.parquet    │ │ test.parquet   │
        │ 721,899 rows   │ │ 90,406 rows    │ │ 90,340 rows    │
        └───────┬────────┘ └────────────────┘ └────────────────┘
                │
                ▼
        ┌────────────────────────────────────────────────────┐
        │ Compute pos_weights on TRAIN ONLY                  │
        │ pos_weight[c] = negatives[c] / positives[c]         │
        └──────────────────────────┬─────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────┐
        │ models/v3_pos_weights.npy                          │
        │ shape: (39,) float32                               │
        │ used by BCEWithLogitsLoss in Day 7                 │
        └────────────────────────────────────────────────────┘
```

### Diagram 2: One paper's labels through the pipeline

```text
Original row from arxiv_cs_clean.parquet

id:        2401.12345
title:     "Efficient Vision Transformers for Medical Images"
abstract:  "We propose..."
cs_cats:   ["cs.CV", "cs.LG", "stat.ML"]
first_cat: "cs.CV"

                            │
                            ▼
        ┌──────────────────────────────────────────────┐
        │ Canonical CS categories only                 │
        │ from first_cat unique values                 │
        │ stat.ML is not part of the CS-only label set │
        └────────────────────┬─────────────────────────┘
                             │
                             ▼
        cs.CV → 11
        cs.LG → 25
        stat.ML → dropped

                            │
                            ▼
        label_indices = [11, 25]

                            │
                            ▼
        saved to parquet as compact sparse labels

                            │
                            ▼
        Day 7 training collator expands it into multi-hot:

        class index:       0  1  2  ... 11 ... 25 ... 38
        target vector:    [0, 0, 0, ... 1 ... 1 ... 0]
```

### Diagram 3: Why the split is done in two stages

The desired final split is:

```text
train = 80%
val   = 10%
test  = 10%
```

The splitter only makes one split at a time, so the script does this:

```text
FULL DATASET: 100%

Step 1: separate test

        ┌───────────────────────────────┐ ┌─────────────┐
        │ trainval = 90%                │ │ test = 10%  │
        └───────────────────────────────┘ └─────────────┘

Step 2: split trainval into train and val

        val must be 10% of original dataset
        trainval is 90% of original dataset

        val_size_within_trainval = 10% / 90% = 11.11%

        ┌──────────────────────────┐ ┌─────────────┐
        │ train = 80% original     │ │ val = 10%   │
        └──────────────────────────┘ └─────────────┘

Final:

        train = 80%
        val   = 10%
        test  = 10%
```

---

## 3. Multi-Label Classification — Deep Dive

### Single-label classification

Single-label means each example belongs to exactly one class.

Example:

```text
paper → cs.LG only
```

The target can be one integer:

```text
label = 25
```

The model usually uses softmax:

```text
39 logits → softmax → 39 probabilities that sum to 1
```

Softmax means categories compete. If `cs.LG` goes up, other categories must go down.

That is correct when labels are mutually exclusive.

### Multi-label classification

Multi-label means each example can belong to more than one class.

Example:

```text
paper → cs.CV + cs.LG
```

The target is not one integer. It is 39 independent yes/no answers:

```text
cs.AI? 0
cs.AR? 0
...
cs.CV? 1
...
cs.LG? 1
...
cs.SY? 0
```

The model should use sigmoid, not softmax:

```text
39 logits → sigmoid independently → 39 yes/no probabilities
```

Why sigmoid?

Because categories should **not compete**.

A paper can be both:

```text
cs.CV = yes
cs.LG = yes
```

Softmax would force the model to choose between them. That teaches the wrong problem.

### What the training target looks like

For 39 categories, every paper has a target vector of length 39:

```text
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ..., 1, ..., 0]
```

This is called a **multi-hot vector**.

It is like one-hot, but with more than one `1` allowed.

---

## 4. Label Indices vs Multi-Hot Vectors

The script stores labels as `label_indices`, not as full multi-hot vectors.

### Multi-hot representation

Suppose there are 39 categories and a paper belongs to `cs.CV` and `cs.LG`.

The multi-hot target might look like:

```text
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ..., 1, ..., 0]
```

This is useful for training.

But it is wasteful on disk.

Most papers have only 1 or 2 labels, so most positions are zero.

Your average labels per paper:

```text
1.52 labels per paper
```

That means each paper has roughly:

```text
1.52 ones
37.48 zeros
```

Storing all those zeros for 902,645 papers is unnecessary.

### Sparse index representation

Instead of storing:

```text
[0, 0, 0, ..., 1, ..., 1, ..., 0]
```

we store only the positions of the `1`s:

```text
[11, 25]
```

That is `label_indices`.

### Why this is smart

The training files stay smaller and easier to inspect:

```text
id | title | abstract | label_indices
```

Then during Day 7 training, the collator expands the sparse indices back into multi-hot vectors batch-by-batch.

So the project uses both representations:

```text
On disk:
label_indices = compact sparse representation

During splitting:
Y = dense multi-hot matrix needed by iterstrat

During training:
labels = dense multi-hot batch tensor needed by BCEWithLogitsLoss
```

### The key distinction

`label_indices` is a storage format.

`multi-hot` is a training/evaluation format.

Same information, different shape.

---

## 5. Why Multi-Label Stratified Split Matters

A naive random split can silently damage your evaluation.

### The danger

Your dataset is imbalanced.

Some categories are very common:

```text
cs.LG: 207,862 train papers
```

Some categories are rare:

```text
cs.OS: 1,036 train papers
```

If you randomly split without caring about labels, common categories will probably be fine. But rare categories can become unstable.

Example:

```text
A rare category has 200 total papers.
Naive random split might put:
train: 190
val: 3
test: 7
```

Now validation has only 3 examples of that class. Macro F1 for that class becomes noisy and unreliable.

Even worse, if a category is very rare, a naive split can put almost all positives in one split.

### Why this matters for your project

You care about **macro F1**.

Macro F1 treats every class equally:

```text
F1(cs.AI)
F1(cs.AR)
...
F1(cs.OS)
...
F1(cs.LG)

macro F1 = average across all 39 classes
```

So rare classes matter a lot.

If the split mishandles rare classes, macro F1 becomes misleading.

### What multi-label stratification tries to preserve

It tries to keep label proportions similar across train/val/test.

Simplified example:

```text
Full dataset:
cs.LG appears in 23% of papers
cs.OS appears in 0.14% of papers

Good split:
train: cs.LG ≈ 23%, cs.OS ≈ 0.14%
val:   cs.LG ≈ 23%, cs.OS ≈ 0.14%
test:  cs.LG ≈ 23%, cs.OS ≈ 0.14%
```

For multi-label data, this is harder than normal stratification because each paper can have several labels.

The splitter must balance many overlapping categories at once.

That is why the script uses:

```python
MultilabelStratifiedShuffleSplit
```

instead of normal `train_test_split`.

---

## 6. Class Imbalance and `pos_weight`

Your classifier has 39 outputs.

For each category, the model answers:

```text
Is this category present? yes/no
```

But most categories are absent for most papers.

Example for a rare class like `cs.OS`:

```text
positive examples: about 1,036
negative examples: about 720,863
```

If you train without class balancing, the model can learn a lazy strategy:

```text
Always predict 0 for cs.OS.
```

That would be correct almost all the time, but useless.

### What `pos_weight` does

The script computes:

```text
pos_weight[c] = negatives_in_class_c / positives_in_class_c
```

For a common class, the weight is small.

Example:

```text
cs.LG pos_weight ≈ 2.5
```

For a rare class, the weight is large.

Example:

```text
cs.OS pos_weight ≈ 695.8
```

This means:

> Missing a positive `cs.OS` example is punished much more strongly than missing a positive `cs.LG` example.

That forces the model to care about rare categories.

### What it does not mean

It does **not** mean `cs.OS` becomes 695 times more important in the final product.

It means training compensates for the fact that rare positives appear much less often.

Without this, the rare class signal would be drowned by millions of negative labels.

### Why `pos_weight` is used in BCEWithLogitsLoss

For multi-label classification, the loss is essentially 39 binary classification losses:

```text
loss for cs.AI
loss for cs.AR
...
loss for cs.SY
```

`pos_weight` modifies the positive part of each class's loss.

Plain English:

```text
If the true label is 1 for a rare class and the model predicts badly,
make that mistake expensive.
```

This is why Day 7 training loads:

```text
models/v3_pos_weights.npy
```

and passes it to:

```python
BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 7. The Leakage Question — Why Train Only

The script computes `pos_weights` on **train only**.

This is not a random detail. It is a correctness rule.

### What would be wrong with using the full dataset?

You might think:

```text
"pos_weight only contains class frequencies.
It does not contain titles, abstracts, or specific test examples.
So why is it leakage?"
```

The answer:

> Because `pos_weight` changes the training loss.

Anything that changes training must be computed from training data only.

If you compute `pos_weight` using train + val + test, then the model is trained with knowledge of the validation/test label distribution.

It learns things like:

```text
cs.OS is this rare in the test set.
cs.LG is this common in the test set.
cs.CR has this exact frequency overall.
```

That information shapes the loss weights.

So the model is not seeing individual test papers, but the training process is still influenced by test-set statistics.

### The exam analogy

Not leakage:

```text
I studied only from the training material.
```

Leakage:

```text
I did not read the test questions,
but I checked how many test answers are A/B/C/D
and adjusted my strategy.
```

That is still test information.

### Correct rule

```text
Training decisions → train data only
Model selection → validation data
Final unbiased evaluation → test data once
```

So the script does:

```python
Y_train = Y[train_idx]
pos_weights = compute_pos_weights(Y_train)
```

not:

```python
pos_weights = compute_pos_weights(Y)
```

### Why this matters in production

The test set is supposed to simulate future unseen papers.

In production, you do not know the exact label distribution of future papers.

If you let the model tune training weights using test distribution, your offline score may be slightly too optimistic.

For portfolio/interview purposes, this detail is strong because it shows you understand evaluation hygiene.

---

## 8. The Script — Section by Section

### File header / docstring

```python
"""
prepare_multilabel_data.py — Build train/val/test splits for multi-label fine-tuning.
...
"""
```

Plain English:

The docstring explains what the script reads, what it writes, and why.

This is important because the script is not just a quick transformation. It defines the training/evaluation foundation for the rest of the project.

The output files become inputs for Day 7:

```text
train.parquet
val.parquet
test.parquet
v3_categories.json
v3_pos_weights.npy
```

If this script is wrong, all downstream training is wrong.

---

### Imports

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
```

Plain English:

* `argparse` lets you pass options from the command line.
* `json` saves the category list.
* `time` is used for timestamps and runtime logging.
* `Path` creates robust file paths.
* `numpy` handles numeric arrays and class weights.
* `pandas` reads/writes parquet files.
* `MultilabelStratifiedShuffleSplit` performs the multi-label-aware split.

The important imported tool is:

```python
MultilabelStratifiedShuffleSplit
```

That is what makes this script different from a naive split script.

---

### Paths

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'arxiv_cs_clean.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
```

Plain English:

The script finds its own location, then builds paths relative to the project root.

This avoids a common bug:

```text
The script works only if I run it from the exact right folder.
```

With this design, you can run:

```bash
python scripts/prepare_multilabel_data.py
```

from the project root, and the script reliably finds files.

Then the script defines all output paths:

```python
CATEGORIES_PATH = MODELS_DIR / 'v3_categories.json'
TRAIN_PATH = PROCESSED_DIR / 'train.parquet'
VAL_PATH = PROCESSED_DIR / 'val.parquet'
TEST_PATH = PROCESSED_DIR / 'test.parquet'
POS_WEIGHTS_PATH = MODELS_DIR / 'v3_pos_weights.npy'
```

Plain English:

These are the artifacts Day 7 depends on.

---

### Helper: `log()`

```python
def log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)
```

Plain English:

Print a message with a timestamp.

Example output:

```text
[22:14:03] Building canonical category list...
```

Why `flush=True`?

Because long-running scripts should show logs immediately. Without flushing, Python may buffer output and the terminal may look frozen.

---

### Helper: `build_label_indices()`

```python
def build_label_indices(cs_cats_list: list, cat_to_idx: dict[str, int]) -> list[int]:
    """Convert a list of category strings to a sorted list of integer indices.
    Drops any category not in cat_to_idx (e.g. cross-listed stat.ML)."""
    if cs_cats_list is None:
        return []
    indices = sorted({cat_to_idx[c] for c in cs_cats_list if c in cat_to_idx})
    return indices
```

Plain English:

Take a paper's category strings and convert them into integer positions.

Example:

```text
cs_cats_list = ["cs.CV", "cs.LG", "stat.ML"]
```

Suppose:

```text
cs.CV → 11
cs.LG → 25
```

and `stat.ML` is not in the CS-only category map.

Then the function returns:

```text
[11, 25]
```

The set comprehension:

```python
{cat_to_idx[c] for c in cs_cats_list if c in cat_to_idx}
```

has two important effects:

1. Drops categories outside the CS label space.
2. Removes duplicates defensively.

Then `sorted(...)` makes the result deterministic.

So `[25, 11]` becomes `[11, 25]`.

Why does sorting matter?

It makes the saved labels consistent and easier to inspect/debug.

---

### Helper: `indices_to_multihot()`

```python
def indices_to_multihot(indices_list: list[list[int]], n_classes: int) -> np.ndarray:
    """Convert a list of index-lists into a dense (n, n_classes) multi-hot matrix.
    Used only for the stratifier — we don't keep this matrix on disk."""
    n = len(indices_list)
    Y = np.zeros((n, n_classes), dtype=np.int8)
    for i, idxs in enumerate(indices_list):
        for j in idxs:
            Y[i, j] = 1
    return Y
```

Plain English:

Convert compact labels into a full 0/1 matrix.

Input:

```text
[
  [11, 25],
  [7],
  [3, 9, 25]
]
```

Output shape if there are 39 classes:

```text
(3, 39)
```

Example rows:

```text
row 0: 1 at columns 11 and 25
row 1: 1 at column 7
row 2: 1 at columns 3, 9, and 25
```

Why does the script build this dense matrix?

Because the stratifier needs a label matrix `Y` to know which papers belong to which classes.

Why not save this dense matrix to disk?

Because it is mostly zeros and unnecessary for storage.

This matrix is temporary infrastructure for splitting and class-frequency calculations.

---

### Helper: `compute_pos_weights()`

```python
def compute_pos_weights(Y_train: np.ndarray) -> np.ndarray:
    """For BCEWithLogitsLoss: pos_weight[c] = (#negatives in class c) / (#positives in class c).
    This up-weights the loss contribution of rare positives so the model doesn't ignore them."""
    n = Y_train.shape[0]
    pos = Y_train.sum(axis=0).astype(np.float64)
    neg = n - pos
    pos_safe = np.where(pos > 0, pos, 1.0)
    weights = neg / pos_safe
    return weights.astype(np.float32)
```

Plain English:

For each category, count:

```text
positives = how many train papers have this category
negatives = how many train papers do not have this category
```

Then compute:

```text
pos_weight = negatives / positives
```

Example:

```text
n = 721,899
positives for cs.OS = 1,036
negatives for cs.OS = 720,863

pos_weight = 720,863 / 1,036 ≈ 695.8
```

That means positive mistakes for `cs.OS` get strongly upweighted during training.

Why `float64` internally?

For stable division.

Why return `float32`?

Because PyTorch training uses float32/float16 style tensors. No need to save class weights as float64.

Why `pos_safe`?

If a class somehow has zero positives in train, division by zero would crash or produce infinity.

The script uses:

```python
pos_safe = np.where(pos > 0, pos, 1.0)
```

This is defensive programming. With proper stratification, every class should have positives, but the script is protected anyway.

---

### `main()` — command-line arguments

```python
parser = argparse.ArgumentParser(description='Prepare multi-label train/val/test splits.')
parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction (default: 0.1)')
parser.add_argument('--test-frac', type=float, default=0.1, help='Test fraction (default: 0.1)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()
```

Plain English:

Let the user configure the split from the terminal.

Default:

```text
train = 80%
val   = 10%
test  = 10%
seed  = 42
```

Usage:

```bash
python scripts/prepare_multilabel_data.py
```

or:

```bash
python scripts/prepare_multilabel_data.py --val-frac 0.1 --test-frac 0.1 --seed 42
```

Why is `seed` important?

Because splitting involves randomness. If you want to reproduce the same train/val/test files later, the same seed should give the same split.

---

### Argument validation

```python
assert 0 < args.val_frac < 1 and 0 < args.test_frac < 1
assert args.val_frac + args.test_frac < 1
```

Plain English:

Reject impossible splits.

Valid:

```text
val = 0.1
test = 0.1
val + test = 0.2
train = 0.8
```

Invalid:

```text
val = 0.6
test = 0.5
val + test = 1.1
train would be negative
```

This is a small safety check, but it prevents silent nonsense.

---

### Step 1: Load corpus

```python
log(f"Loading {DATA_PATH.name}...")
df = pd.read_parquet(DATA_PATH)
log(f"  {len(df):,} papers, columns: {list(df.columns)}")
```

Plain English:

Read the cleaned arXiv CS dataset into a pandas DataFrame.

Expected shape:

```text
902,645 rows
9 columns
```

Important columns:

```text
id
title
abstract
cs_cats
first_cat
```

Then the script validates required columns:

```python
required = {'id', 'title', 'abstract', 'cs_cats', 'first_cat'}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {missing}")
```

Plain English:

Fail early if the input dataset is not what the script expects.

This is better than crashing later with a confusing error.

---

### Step 2: Build canonical category list

```python
categories = sorted(df['first_cat'].dropna().unique().tolist())
n_classes = len(categories)
cat_to_idx = {c: i for i, c in enumerate(categories)}
```

Plain English:

Create the official list of 39 CS categories.

The source of truth is:

```text
first_cat
```

Why `first_cat`?

Because it is already known to contain the 39 CS categories used in V1.

Why not use `cs_cats` directly?

Because `cs_cats` can include cross-listed non-CS categories like:

```text
stat.ML
math.OC
eess.IV
```

But this classifier's label space is CS-only.

So the canonical category list comes from `first_cat`, and `cs_cats` is filtered against that list.

The mapping looks like:

```text
cs.AI → 0
cs.AR → 1
...
cs.SY → 38
```

Then it saves the list:

```python
with open(CATEGORIES_PATH, 'w') as f:
    json.dump(categories, f, indent=2)
```

Why save this?

Because Day 7 and inference need to know what each output index means.

The model outputs:

```text
39 logits
```

Without `v3_categories.json`, index `25` is just a number. With the JSON file, index `25` becomes a category like `cs.LG`.

---

### Step 3: Build `label_indices`

```python
df['label_indices'] = df['cs_cats'].apply(lambda x: build_label_indices(x, cat_to_idx))
```

Plain English:

For every paper, convert its category strings into integer indices.

Example:

```text
cs_cats = ["cs.CV", "cs.LG", "stat.ML"]
```

becomes:

```text
label_indices = [11, 25]
```

Then the script defensively drops papers with zero labels:

```python
df = df[df['label_indices'].apply(len) > 0].reset_index(drop=True)
```

This should not happen after previous cleaning, but it protects the pipeline.

Then it logs average labels per paper:

```python
avg_labels = df['label_indices'].apply(len).mean()
```

Your result:

```text
Avg labels per paper: 1.52
```

This is important because it proves multi-label is real.

Most papers are single-label, but enough papers have multiple CS labels to justify a multi-label classifier.

---

### Step 4: Build temporary multi-hot matrix

```python
Y = indices_to_multihot(df['label_indices'].tolist(), n_classes)
X_idx = np.arange(len(df))[:, None]
```

Plain English:

Build:

```text
Y shape = (902,645, 39)
```

Each row is the multi-hot label vector for one paper.

Example:

```text
paper 0 → [0, 0, ..., 1, ..., 1, ..., 0]
paper 1 → [0, 1, ..., 0]
...
```

Why `X_idx`?

The splitter expects both `X` and `y`, but for splitting we do not need real input features.

We only need labels.

So the script gives it placeholder indices:

```python
X_idx = np.arange(len(df))[:, None]
```

This means:

```text
paper 0
paper 1
paper 2
...
```

The stratifier uses `Y` to balance labels and returns row indices.

---

### Step 5: Multi-label stratified split

```python
msss_test = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=args.test_frac, random_state=args.seed,
)
trainval_idx, test_idx = next(msss_test.split(X_idx, Y))
```

Plain English:

First split off the test set.

With `test_frac=0.1`:

```text
trainval = 90%
test = 10%
```

Then split trainval into train and validation:

```python
val_size_within = args.val_frac / (1.0 - args.test_frac)
```

This line is easy to miss, but important.

We want validation to be 10% of the original dataset, not 10% of the remaining 90%.

So:

```text
val_size_within = 0.1 / 0.9 = 0.1111
```

Then:

```python
msss_val = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=val_size_within, random_state=args.seed,
)
train_idx_rel, val_idx_rel = next(msss_val.split(trainval_idx[:, None], Y[trainval_idx]))
train_idx = trainval_idx[train_idx_rel]
val_idx = trainval_idx[val_idx_rel]
```

Plain English:

Split the 90% trainval pool into:

```text
train = 80% original dataset
val   = 10% original dataset
```

The splitter returns indices relative to `trainval_idx`, so the script maps them back to original dataset indices:

```python
train_idx = trainval_idx[train_idx_rel]
val_idx = trainval_idx[val_idx_rel]
```

This is a common source of bugs. The script handles it correctly.

---

### Sanity checks

```python
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0
assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)
```

Plain English:

Verify:

1. Train and val do not overlap.
2. Train and test do not overlap.
3. Val and test do not overlap.
4. Every paper appears in exactly one split.

This prevents catastrophic evaluation leakage.

If the same paper appears in both train and test, your test score becomes inflated.

These assertions guarantee split integrity.

---

### Step 6: Save splits

```python
keep_cols = ['id', 'title', 'abstract', 'label_indices']
```

Plain English:

Only keep the columns needed for training.

Why not save all original columns?

Because Day 7 needs only:

```text
id: for traceability
title: model input
abstract: model input
label_indices: target labels
```

Then:

```python
df.iloc[train_idx][keep_cols].to_parquet(TRAIN_PATH, index=False)
df.iloc[val_idx][keep_cols].to_parquet(VAL_PATH, index=False)
df.iloc[test_idx][keep_cols].to_parquet(TEST_PATH, index=False)
```

Plain English:

Write three parquet files.

These are the actual training/evaluation datasets for the LoRA classifier.

---

### Step 7: Compute `pos_weights` on train only

```python
Y_train = Y[train_idx]
pos_weights = compute_pos_weights(Y_train)
np.save(POS_WEIGHTS_PATH, pos_weights)
```

Plain English:

Take only the train rows of the label matrix.

Compute class imbalance weights from train labels only.

Save them as:

```text
models/v3_pos_weights.npy
```

Day 7 loads this file and passes it to:

```python
BCEWithLogitsLoss(pos_weight=pos_weight)
```

This is one of the most important outputs of Day 6.

---

### Diagnostics: rarest and most common categories

```python
train_pos = Y_train.sum(axis=0)
rarest = np.argsort(train_pos)[:3]
commonest = np.argsort(train_pos)[-3:][::-1]
```

Plain English:

Count positives per class in train.

Then show:

```text
3 rarest categories
3 most common categories
```

Example from your run:

```text
Rarest:
cs.OS      1,036 papers  pos_weight=695.8

Most common:
cs.LG    207,862 papers  pos_weight=2.5
```

This diagnostic confirms that class imbalance is real and severe.

It also helps you explain why `pos_weight` is not optional.

---

## 9. Why Specific Engineering Choices

### Why use `first_cat` to build category list?

Because your classifier's label space is CS-only.

`first_cat` has the canonical 39 CS categories.

`cs_cats` may include cross-listed non-CS categories.

If you built the label space from `cs_cats` directly, you could accidentally expand the classifier beyond CS into categories like:

```text
stat.ML
math.OC
eess.IV
```

That would change the problem.

The script correctly defines:

```text
Label space = 39 CS categories
```

and filters multi-label targets into that space.

---

### Why save `v3_categories.json`?

The model outputs numbers, not names.

Example:

```text
logits shape = (batch_size, 39)
```

Output position 25 means nothing unless you know:

```text
categories[25] = "cs.LG"
```

So `v3_categories.json` is the dictionary that maps model outputs back to human-readable categories.

Without it, inference is impossible to interpret safely.

---

### Why store `label_indices` instead of multi-hot vectors?

Because the dataset has 902,645 rows and 39 categories.

A dense label matrix has:

```text
902,645 × 39 = 35,203,155 positions
```

Most are zeros.

Average labels per paper is only:

```text
1.52
```

So sparse storage is much cleaner:

```text
[11, 25]
```

instead of:

```text
[0, 0, 0, 0, ..., 1, ..., 1, ..., 0]
```

The training script can reconstruct dense multi-hot labels batch-by-batch.

---

### Why build dense `Y` temporarily?

Because the stratifier needs a matrix describing which classes each example belongs to.

The dense matrix is used for:

```text
multi-label stratified splitting
class frequency counts
pos_weight calculation
```

But it is not saved permanently.

This is a good compromise:

```text
Use dense when algorithms need it.
Store sparse when humans/files need it.
```

---

### Why use multi-label stratification instead of random split?

Because rare categories can be mishandled by naive random splits.

For macro F1, every class matters equally.

If a rare category has too few validation/test examples, your metric becomes noisy.

Multi-label stratification keeps label proportions more stable across splits.

This makes evaluation more trustworthy.

---

### Why split test first, then validation?

Because the splitter only performs one train/test split at a time.

The script wants:

```text
80 / 10 / 10
```

So it does:

```text
100% → 90% trainval + 10% test
90% trainval → 80% train + 10% val
```

The key formula:

```python
val_size_within = val_frac / (1.0 - test_frac)
```

For 10% val and 10% test:

```text
0.1 / 0.9 = 0.1111
```

So validation is 11.11% of the trainval pool, which equals 10% of the original dataset.

---

### Why compute `pos_weights` on train only?

Because `pos_weights` affects training.

If you compute it from full data, the training loss knows validation/test class frequencies.

That is leakage.

Correct:

```python
Y_train = Y[train_idx]
pos_weights = compute_pos_weights(Y_train)
```

Wrong:

```python
pos_weights = compute_pos_weights(Y)
```

This is a small detail with big evaluation meaning.

---

### Why save as `.npy`?

`pos_weights` is just a numeric vector:

```text
shape = (39,)
dtype = float32
```

NumPy `.npy` is simple, fast, and preserves exact shape/dtype.

The training script can load it directly:

```python
pos_weights_np = np.load(POS_WEIGHTS_PATH).astype(np.float32)
```

---

### Why use parquet for train/val/test?

Parquet is efficient for tabular data.

It preserves columns like:

```text
id
title
abstract
label_indices
```

It is faster and smaller than CSV for this kind of dataset.

It also preserves list-like columns better than plain CSV.

---

## 10. What to Whiteboard

Structure the explanation in four parts.

### Section A — Problem transformation

Draw:

```text
V1:
paper → one category

V3:
paper → multiple categories
```

Then explain:

```text
This requires multi-label targets, sigmoid outputs, and BCEWithLogitsLoss.
```

### Section B — Label encoding

Draw one example:

```text
cs_cats = ["cs.CV", "cs.LG", "stat.ML"]

canonical CS-only label space:
cs.CV → 11
cs.LG → 25

stat.ML dropped

label_indices = [11, 25]

training multi-hot:
[0, 0, ..., 1, ..., 1, ..., 0]
```

Key sentence:

> We store sparse indices on disk to avoid millions of zeros, then expand to multi-hot during training.

### Section C — Split strategy

Draw:

```text
Full dataset 100%
        │
        ├── test 10%
        └── trainval 90%
                │
                ├── val 10% original
                └── train 80% original
```

Then explain:

```text
We use multi-label stratification so rare category proportions survive across train, validation, and test.
```

### Section D — Class imbalance

Draw:

```text
cs.LG: many positives → small pos_weight
cs.OS: few positives  → huge pos_weight
```

Formula:

```text
pos_weight[c] = negatives[c] / positives[c]
```

Then say:

> This is computed on train only because it changes the training loss. Using validation/test frequencies would leak evaluation-set information into training.

---

## 11. What Makes This Script Professional

If an interviewer opens this file and asks you to explain it, these are the professional signals:

### 1. It creates reproducible training artifacts

The script has clear inputs and outputs:

```text
input: full cleaned corpus
outputs: train/val/test, categories, pos_weights
```

This is not an ad hoc notebook cell. It is a reproducible data-preparation step.

### 2. It uses a canonical label space

The script explicitly defines the 39-category CS label space and saves it.

This prevents category-index mismatch later.

### 3. It handles multi-label targets correctly

It does not force papers into one label.

It preserves multiple CS categories per paper.

This aligns the data with the real task.

### 4. It uses sparse storage intelligently

It stores `label_indices` instead of dense multi-hot vectors.

That keeps training files compact and readable.

### 5. It uses multi-label stratification

This shows awareness that normal random splitting is not enough for imbalanced multi-label data.

### 6. It protects against data leakage

`pos_weights` are computed from train only.

That demonstrates proper evaluation hygiene.

### 7. It validates assumptions

The script checks required columns, split overlap, and full coverage.

These checks prevent silent data corruption.

### 8. It logs useful diagnostics

It prints:

```text
number of papers
categories
average labels per paper
split sizes
pos_weight range
rarest/commonest categories
```

These logs make it easier to trust the output.

### 9. It is configurable from the CLI

You can change validation fraction, test fraction, and seed without editing the file.

This is maintainable engineering.

### 10. It produces exactly what Day 7 needs

The output is aligned with the LoRA training script:

```text
train/val/test parquet → datasets
categories.json → output interpretation
pos_weights.npy → BCEWithLogitsLoss
```

That shows pipeline thinking.

---

## 12. Quick Reference Card

### Numbers to remember

```text
Full corpus:              902,645 papers
Train split:              721,899 papers
Validation split:          90,406 papers
Test split:                90,340 papers
Number of CS categories:       39
Average labels per paper:     1.52
Most common train class:    cs.LG, 207,862 papers
Rarest train class:         cs.OS, 1,036 papers
pos_weight range:           2.5 to 695.8
```

### Core files

```text
Input:
data/processed/arxiv_cs_clean.parquet

Outputs:
models/v3_categories.json
data/processed/train.parquet
data/processed/val.parquet
data/processed/test.parquet
models/v3_pos_weights.npy
```

### Core formulas

Multi-hot target:

```text
label_indices = [11, 25]
→ vector length 39 with 1s at positions 11 and 25
```

Class imbalance weight:

```text
pos_weight[c] = negatives_in_train_class_c / positives_in_train_class_c
```

Validation size inside trainval:

```text
val_size_within = val_frac / (1 - test_frac)
```

For 80/10/10:

```text
val_size_within = 0.1 / 0.9 = 0.1111
```

### Key concepts

```text
Multi-label classification:
A paper can have several categories.

Multi-hot vector:
Length-39 target vector with multiple 1s allowed.

Label indices:
Compact sparse representation of multi-hot labels.

Multi-label stratification:
Split method that preserves label proportions across train/val/test.

pos_weight:
Loss weighting that makes rare positive labels matter during training.

Leakage:
Any training-time decision using validation/test information.
```

### Pipeline

```text
full corpus
→ canonical 39-category list
→ cs_cats converted to label_indices
→ temporary multi-hot matrix Y
→ multi-label stratified 80/10/10 split
→ save train/val/test parquet
→ compute train-only pos_weights
→ Day 7 LoRA fine-tuning
```

### Interview explanation in 30 seconds

> I prepared the arXiv dataset for multi-label classification because papers can belong to multiple CS categories. I built a canonical 39-category CS label space, converted each paper's `cs_cats` into sparse `label_indices`, used multi-label stratified splitting to preserve rare-class proportions across train, validation, and test, and computed `pos_weight` on the training split only to handle class imbalance without leaking validation/test statistics into training. The outputs feed directly into the LoRA fine-tuning script: train/val/test parquet files, category mapping, and BCE loss weights.

### Interview explanation in 2 minutes

> The goal of this script is to convert the cleaned arXiv corpus into reliable inputs for multi-label fine-tuning. The raw dataset has `cs_cats`, which can contain multiple categories per paper, so instead of forcing a single label like in V1, I preserve all CS labels. I define the official 39-class label space from `first_cat`, then filter `cs_cats` into that space and convert labels into integer indices. I store those sparse indices on disk because dense multi-hot vectors would mostly be zeros.
>
> For splitting, I build a temporary dense multi-hot matrix and use `MultilabelStratifiedShuffleSplit`, because normal random splitting can break rare categories and make macro F1 unreliable. I split test first, then split the remaining trainval pool so the final proportions are 80/10/10. I also assert that the splits have no overlap and cover the full dataset.
>
> Finally, I compute `pos_weight` from the training labels only. This gives BCEWithLogitsLoss a per-class imbalance correction, so rare categories like `cs.OS` are not ignored. Computing it on train only avoids leakage, because class frequencies influence the training loss. The final artifacts are `train.parquet`, `val.parquet`, `test.parquet`, `v3_categories.json`, and `v3_pos_weights.npy`, which are exactly what the LoRA training script needs.
