"""
  1. Runs inference on the validation set to collect per-class probabilities
  2. Sweeps thresholds independently per class, maximizing per-class F1
  3. Saves the 39 optimal thresholds to a JSON file
  4. Re-evaluates on the test set using the optimized thresholds
  5. Compares global vs per-class threshold performance
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss


# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("models/v3_lora_classifier_r32_alpha64_len384_batch64_expI/best_micro")
VAL_PATH = Path("data/processed/val.parquet")
TEST_PATH = Path("data/processed/test.parquet")
CATEGORIES_PATH = Path("models/v3_categories.json")
OUTPUT_DIR = Path("models/v3_lora_classifier_r32_alpha64_len384_batch64_expI/test_results")
CACHE_DIR = OUTPUT_DIR / "cached_predictions"

BATCH_SIZE = 512
MAX_LENGTH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SWEEP_RANGE = np.arange(0.05, 0.96, 0.01)


# ── Model ────────────────────────────────────────────────────────────────────

class Specter2LoRAClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(0.0)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


class PaperDataset(Dataset):
    def __init__(self, df, tokenizer, num_labels, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['title']} [SEP] {row['abstract']}"
        enc = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        target = torch.zeros(self.num_labels, dtype=torch.float32)
        for li in row["label_indices"]:
            target[li] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": target,
        }


def load_model(checkpoint_dir, num_labels):
    with open(checkpoint_dir / "training_config.json") as f:
        config = json.load(f)
    base = config.get("model_name", "allenai/specter2_base")
    model = Specter2LoRAClassifier(base, num_labels)
    model.encoder = PeftModel.from_pretrained(model.encoder, checkpoint_dir / "adapter")
    model.encoder = model.encoder.merge_and_unload()
    state = torch.load(checkpoint_dir / "classifier.pt", map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    return model, tokenizer


def run_inference(model, tokenizer, df, num_labels, label="dataset"):
    """Run inference and return (probabilities, targets) as numpy arrays."""
    dataset = PaperDataset(df, tokenizer, num_labels, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs, all_targets = [], []
    t0 = time.time()

    print(f"\n  Running inference on {len(df):,} papers ({label})...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            logits = model(
                batch["input_ids"].to(DEVICE),
                batch["attention_mask"].to(DEVICE),
            )
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(batch["target"].numpy())
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                done = (i + 1) * BATCH_SIZE
                print(f"    {done:,}/{len(df):,} | {done/elapsed:.0f} papers/sec")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({len(df)/elapsed:.0f} papers/sec)")

    return np.concatenate(all_probs), np.concatenate(all_targets)

# ── Threshold optimization ───────────────────────────────────────────────────

def optimize_per_class_thresholds(probs, targets, categories):
    """Find the threshold that maximizes F1 for each class independently."""
    num_classes = len(categories)
    thresholds = np.zeros(num_classes)

    print("\n  Per-class threshold optimization:")
    print(f"  {'Category':<10} {'Threshold':>10} {'F1':>8} {'Prec':>8} {'Recall':>8}")
    print("  " + "-" * 50)

    for c in range(num_classes):
        y_true = targets[:, c]
        p = probs[:, c]

        best_f1, best_t = 0.0, 0.50
        for t in SWEEP_RANGE:
            y_pred = (p >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[c] = best_t
        y_pred = (p >= best_t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        print(f"  {categories[c]:<10} {best_t:>10.2f} {best_f1:>8.3f} {prec:>8.3f} {rec:>8.3f}")

    return thresholds

def evaluate(probs, targets, thresholds, categories, label=""):
    """Evaluate with given thresholds (global or per-class)."""
    if isinstance(thresholds, (float, int)):
        preds = (probs >= thresholds).astype(int)
        threshold_label = f"global={thresholds}"
    else:
        preds = (probs >= thresholds[np.newaxis, :]).astype(int)
        threshold_label = "per-class"

    # Fallback: assign top-1 if no prediction
    empty_mask = preds.sum(axis=1) == 0
    if empty_mask.any():
        top1 = probs[empty_mask].argmax(axis=1)
        for row, col in zip(np.where(empty_mask)[0], top1):
            preds[row, col] = 1

    macro = f1_score(targets, preds, average="macro", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    weighted = f1_score(targets, preds, average="weighted", zero_division=0)
    h_loss = hamming_loss(targets, preds)
    avg_labels = preds.sum(axis=1).mean()

    print(f"\n  [{label}] Thresholds: {threshold_label}")
    print(f"    Macro F1:     {macro:.4f}")
    print(f"    Micro F1:     {micro:.4f}")
    print(f"    Weighted F1:  {weighted:.4f}")
    print(f"    Hamming Loss: {h_loss:.4f}")
    print(f"    Avg labels:   {avg_labels:.2f} (true: {targets.sum(axis=1).mean():.2f})")
    print(f"    Empty (before fallback): {empty_mask.sum()} ({empty_mask.mean():.1%})")

    return {
        "macro_f1": float(round(macro, 4)),
        "micro_f1": float(round(micro, 4)),
        "weighted_f1": float(round(weighted, 4)),
        "hamming_loss": float(round(h_loss, 4)),
        "avg_labels": float(round(avg_labels, 2)),
        "empty_before_fallback": int(empty_mask.sum()),
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip inference, load cached predictions")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(CATEGORIES_PATH) as f:
        categories = json.load(f)
    num_labels = len(categories)

    # ── Load or run validation inference ─────────────────────────────────
    val_probs_path = CACHE_DIR / "val_probs.npy"
    val_targets_path = CACHE_DIR / "val_targets.npy"

    if args.eval_only and val_probs_path.exists():
        print("Loading cached validation predictions...")
        val_probs = np.load(val_probs_path)
        val_targets = np.load(val_targets_path)
    else:
        print("Loading model...")
        model, tokenizer = load_model(CHECKPOINT_DIR, num_labels)
        model.to(DEVICE).eval()

        val_df = pd.read_parquet(VAL_PATH)
        val_probs, val_targets = run_inference(
            model, tokenizer, val_df, num_labels, label="validation"
        )
        np.save(val_probs_path, val_probs)
        np.save(val_targets_path, val_targets)
        print(f"  Cached to {CACHE_DIR}/")

    # ── Optimize thresholds on validation set ────────────────────────────
    print("\n" + "=" * 60)
    print("OPTIMIZING PER-CLASS THRESHOLDS (validation set)")
    print("=" * 60)

    per_class_t = optimize_per_class_thresholds(val_probs, val_targets, categories)

    # Save thresholds
    thresholds_dict = {categories[i]: float(round(per_class_t[i], 2))
                       for i in range(num_labels)}
    thresholds_path = OUTPUT_DIR / "per_class_thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f"\n  Saved thresholds to {thresholds_path}")

    # Compare on validation set
    print("\n" + "=" * 60)
    print("VALIDATION SET COMPARISON")
    print("=" * 60)
    evaluate(val_probs, val_targets, 0.90, categories, label="Global 0.90")
    evaluate(val_probs, val_targets, per_class_t, categories, label="Per-class")

    # ── Evaluate on test set ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_probs_path = CACHE_DIR / "test_probs.npy"
    test_targets_path = CACHE_DIR / "test_targets.npy"

    if args.eval_only and test_probs_path.exists():
        print("Loading cached test predictions...")
        test_probs = np.load(test_probs_path)
        test_targets = np.load(test_targets_path)
    else:
        if "model" not in dir() or model is None:
            print("Loading model...")
            model, tokenizer = load_model(CHECKPOINT_DIR, num_labels)
            model.to(DEVICE).eval()

        test_df = pd.read_parquet(TEST_PATH)
        test_probs, test_targets = run_inference(
            model, tokenizer, test_df, num_labels, label="test"
        )
        np.save(test_probs_path, test_probs)
        np.save(test_targets_path, test_targets)

    global_results = evaluate(test_probs, test_targets, 0.90, categories,
                              label="Global 0.90")
    perclass_results = evaluate(test_probs, test_targets, per_class_t, categories,
                                label="Per-class")

    preds_opt = (test_probs >= per_class_t[np.newaxis, :]).astype(int)
    empty = preds_opt.sum(axis=1) == 0
    if empty.any():
        for row in np.where(empty)[0]:
            preds_opt[row, test_probs[row].argmax()] = 1

    per_class_f1 = f1_score(test_targets, preds_opt, average=None, zero_division=0)
    per_class_p = precision_score(test_targets, preds_opt, average=None, zero_division=0)
    per_class_r = recall_score(test_targets, preds_opt, average=None, zero_division=0)
    support = test_targets.sum(axis=0).astype(int)

    class_df = pd.DataFrame({
        "category": categories,
        "f1": per_class_f1,
        "precision": per_class_p,
        "recall": per_class_r,
        "support": support,
        "threshold": per_class_t,
    }).sort_values("f1", ascending=False)

    print(f"\n  {'Category':<10} {'Thresh':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Support':>9}")
    print("  " + "-" * 52)
    for _, row in class_df.iterrows():
        print(f"  {row['category']:<10} {row['threshold']:>7.2f} {row['f1']:>7.3f} "
              f"{row['precision']:>7.3f} {row['recall']:>7.3f} {row['support']:>9,}")

    class_df.to_csv(OUTPUT_DIR / "per_class_metrics_optimized.csv", index=False)

    # Summary
    delta_macro = perclass_results["macro_f1"] - global_results["macro_f1"]
    print(f"\n  Macro F1 improvement: {global_results['macro_f1']:.4f} → "
          f"{perclass_results['macro_f1']:.4f} ({delta_macro:+.4f})")

    summary = {
        "thresholds": thresholds_dict,
        "test_global_090": global_results,
        "test_per_class": perclass_results,
        "macro_f1_delta": float(round(delta_macro, 4)),
    }
    with open(OUTPUT_DIR / "threshold_optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
