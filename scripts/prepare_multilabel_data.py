from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ---------- Paths ----------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'arxiv_cs_clean.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

CATEGORIES_PATH = MODELS_DIR / 'v3_categories.json'
TRAIN_PATH = PROCESSED_DIR / 'train.parquet'
VAL_PATH = PROCESSED_DIR / 'val.parquet'
TEST_PATH = PROCESSED_DIR / 'test.parquet'
POS_WEIGHTS_PATH = MODELS_DIR / 'v3_pos_weights.npy'


# ---------- Helpers ----------

def log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def build_label_indices(cs_cats_list: list, cat_to_idx: dict[str, int]) -> list[int]:
    """Convert a list of category strings to a sorted list of integer indices.
    Drops any category not in cat_to_idx (e.g. cross-listed stat.ML)."""
    if cs_cats_list is None:
        return []
    indices = sorted({cat_to_idx[c] for c in cs_cats_list if c in cat_to_idx})
    return indices


def indices_to_multihot(indices_list: list[list[int]], n_classes: int) -> np.ndarray:
    """Convert a list of index-lists into a dense (n, n_classes) multi-hot matrix.
    Used only for the stratifier — we don't keep this matrix on disk."""
    n = len(indices_list)
    Y = np.zeros((n, n_classes), dtype=np.int8)
    for i, idxs in enumerate(indices_list):
        for j in idxs:
            Y[i, j] = 1
    return Y


def compute_pos_weights(Y_train: np.ndarray) -> np.ndarray:
    """For BCEWithLogitsLoss: pos_weight[c] = (#negatives in class c) / (#positives in class c).
    This up-weights the loss contribution of rare positives so the model doesn't ignore them."""
    n = Y_train.shape[0]
    pos = Y_train.sum(axis=0).astype(np.float64)         # positives per class
    neg = n - pos                                          # negatives per class
    # avoid divide-by-zero: if a class has no positives in train (shouldn't happen with stratification),
    # set its weight to 1.0 so it's effectively neutral
    pos_safe = np.where(pos > 0, pos, 1.0)
    weights = neg / pos_safe
    return weights.astype(np.float32)


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare multi-label train/val/test splits.')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Validation fraction (default: 0.1)')
    parser.add_argument('--test-frac', type=float, default=0.1, help='Test fraction (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    assert 0 < args.val_frac < 1 and 0 < args.test_frac < 1
    assert args.val_frac + args.test_frac < 1

    overall_t0 = time.time()

    # ---------- 1. Load corpus ----------
    log(f"Loading {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    log(f"  {len(df):,} papers, columns: {list(df.columns)}")

    required = {'id', 'title', 'abstract', 'cs_cats', 'first_cat'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # ---------- 2. Canonical category list ----------
    # We use first_cat (already known to have 39 unique values) as the source of truth.
    # cs_cats may contain extra cross-listed categories from outside CS (e.g. stat.ML, math.OC) —
    # those are dropped because the classifier's label space is CS-only.
    log("Building canonical category list...")
    categories = sorted(df['first_cat'].dropna().unique().tolist())
    n_classes = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    log(f"  {n_classes} categories: {categories[:5]}... {categories[-3:]}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATEGORIES_PATH, 'w') as f:
        json.dump(categories, f, indent=2)
    log(f"  Saved → {CATEGORIES_PATH.name}")

    # ---------- 3. Build label_indices column ----------
    log("Converting cs_cats → label_indices (sparse representation)...")
    t0 = time.time()
    df['label_indices'] = df['cs_cats'].apply(lambda x: build_label_indices(x, cat_to_idx))
    log(f"  Done in {time.time()-t0:.1f}s")

    # Drop any paper that ended up with zero CS labels (defensive — shouldn't happen given V1 cleaning)
    n_before = len(df)
    df = df[df['label_indices'].apply(len) > 0].reset_index(drop=True)
    if len(df) < n_before:
        log(f"  Dropped {n_before - len(df):,} papers with no CS labels after filtering")

    # Diagnostic: average labels per paper
    avg_labels = df['label_indices'].apply(len).mean()
    log(f"  Avg labels per paper: {avg_labels:.2f}")

    # ---------- 4. Build dense multi-hot only for the stratifier ----------
    log("Building temporary dense multi-hot matrix for stratifier (in-memory only)...")
    Y = indices_to_multihot(df['label_indices'].tolist(), n_classes)
    X_idx = np.arange(len(df))[:, None]   # placeholder X; stratifier only needs y to split

    # ---------- 5. Multi-label stratified split: 80/10/10 ----------
    # MultilabelStratifiedShuffleSplit does ONE split at a time.
    # Strategy: first separate test (test_frac), then split the remainder into train/val.
    log(f"Stratified split: train / val ({args.val_frac:.0%}) / test ({args.test_frac:.0%})...")

    # Step 1: peel off test
    msss_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.test_frac, random_state=args.seed,
    )
    trainval_idx, test_idx = next(msss_test.split(X_idx, Y))

    # Step 2: split remainder into train and val.
    # We want val to end up as args.val_frac of the ORIGINAL dataset, so within the
    # trainval pool it should be val_frac / (1 - test_frac).
    val_size_within = args.val_frac / (1.0 - args.test_frac)
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_size_within, random_state=args.seed,
    )
    train_idx_rel, val_idx_rel = next(msss_val.split(trainval_idx[:, None], Y[trainval_idx]))
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]

    log(f"  train: {len(train_idx):,}  val: {len(val_idx):,}  test: {len(test_idx):,}")

    # Sanity check: no overlap, full coverage
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    # ---------- 6. Save splits ----------
    keep_cols = ['id', 'title', 'abstract', 'label_indices']
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Saving train → {TRAIN_PATH.name}...")
    df.iloc[train_idx][keep_cols].to_parquet(TRAIN_PATH, index=False)
    log(f"Saving val   → {VAL_PATH.name}...")
    df.iloc[val_idx][keep_cols].to_parquet(VAL_PATH, index=False)
    log(f"Saving test  → {TEST_PATH.name}...")
    df.iloc[test_idx][keep_cols].to_parquet(TEST_PATH, index=False)

    # ---------- 7. Compute pos_weights (TRAIN ONLY — no leakage) ----------
    log("Computing pos_weights on TRAIN ONLY (used by BCEWithLogitsLoss to handle class imbalance)...")
    Y_train = Y[train_idx]
    pos_weights = compute_pos_weights(Y_train)
    np.save(POS_WEIGHTS_PATH, pos_weights)
    log(f"  Saved → {POS_WEIGHTS_PATH.name}")
    log(f"  Min weight: {pos_weights.min():.1f}  Max weight: {pos_weights.max():.1f}  Median: {np.median(pos_weights):.1f}")

    # Show the 3 rarest and 3 most common categories in train
    train_pos = Y_train.sum(axis=0)
    rarest = np.argsort(train_pos)[:3]
    commonest = np.argsort(train_pos)[-3:][::-1]
    log("  Rarest in train:")
    for i in rarest:
        log(f"    {categories[i]:12s}  {train_pos[i]:>6,} papers  pos_weight={pos_weights[i]:.1f}")
    log("  Most common in train:")
    for i in commonest:
        log(f"    {categories[i]:12s}  {train_pos[i]:>6,} papers  pos_weight={pos_weights[i]:.1f}")

    log("─" * 60)
    log(f"✓ Done in {time.time()-overall_t0:.1f}s")


if __name__ == '__main__':
    main()
