import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ---------- Config ----------
CHECKPOINT_DIR = Path("models/v3_lora_classifier_r32_alpha64_len384_batch64_expI/best_micro")
TEST_PATH = Path("data/processed/test.parquet")
FULL_DATA_PATH = Path("data/processed/arxiv_cs_clean.parquet")
CATEGORIES_PATH = Path("models/v3_categories.json")
THRESHOLD = 0.90
BATCH_SIZE = 64
MAX_LENGTH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("models/v3_lora_classifier_r32_alpha64_len384_batch64_expI/test_results")

# ---------- JSON helpers ----------

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    tmp_path.replace(path)

# ---------- Model architecture ----------

class Specter2LoRAClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

# ---------- Dataset ----------

class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, categories: list[str], max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.num_labels = len(categories)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = f"{row['title']} [SEP] {row['abstract']}"

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = torch.zeros(self.num_labels, dtype=torch.float32)
        for li in row["label_indices"]:
            target[int(li)] = 1.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": target,
        }

# ---------- Load model ----------

def load_model(checkpoint_dir: Path, num_labels: int) -> tuple[nn.Module, AutoTokenizer]:
    config_path = checkpoint_dir / "training_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_model_name = config.get("model_name", "allenai/specter2_base")

    model = Specter2LoRAClassifier(base_model_name, num_labels, dropout=0.0)

    model.encoder = PeftModel.from_pretrained(model.encoder, checkpoint_dir / "adapter")
    model.encoder = model.encoder.merge_and_unload()

    classifier_state = torch.load(
        checkpoint_dir / "classifier.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.classifier.load_state_dict(classifier_state)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    return model, tokenizer

# ---------- Main ----------

def main() -> None:
    print("=" * 70)
    print("Test-set evaluation")
    print("=" * 70)

    # Load categories.
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories = json.load(f)

    num_labels = len(categories)
    print(f"Categories: {num_labels}")

    # Load test data.
    print(f"Loading test set from {TEST_PATH}...")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"Test set: {len(test_df):,} papers")

    # Load first_cat mapping for V1 comparison.
    print("Loading full data for first_cat mapping...")
    full_df = pd.read_parquet(FULL_DATA_PATH, columns=["id", "first_cat"])
    first_cat_map = dict(zip(full_df["id"], full_df["first_cat"]))
    del full_df

    # Map first_cat to index.
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    test_df["first_cat"] = test_df["id"].map(first_cat_map)
    test_df["first_cat_idx"] = test_df["first_cat"].map(cat_to_idx)

    before = len(test_df)
    test_df = test_df.dropna(subset=["first_cat_idx"])
    test_df["first_cat_idx"] = test_df["first_cat_idx"].astype(int)

    if len(test_df) < before:
        print(f"  Dropped {before - len(test_df)} papers with unknown first_cat")

    # Load model.
    print(f"Loading model from {CHECKPOINT_DIR}...")
    model, tokenizer = load_model(CHECKPOINT_DIR, num_labels)
    model.to(DEVICE)
    model.eval()
    print(f"Device: {DEVICE}")

    # DataLoader.
    dataset = TestDataset(test_df, tokenizer, categories, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Inference.
    print(f"\nRunning inference on {len(test_df):,} papers (batch_size={BATCH_SIZE})...")

    all_probs = []
    all_targets = []

    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["target"]

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets.numpy())

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                done = min((i + 1) * BATCH_SIZE, len(test_df))
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(test_df) - done) / rate if rate > 0 else 0.0

                print(
                    f"  Batch {i + 1}/{len(loader)} | "
                    f"{done:,}/{len(test_df):,} | "
                    f"{rate:.0f} papers/sec | ETA {eta:.0f}s"
                )

    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s ({len(test_df) / elapsed:.0f} papers/sec)")

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Multi-label metrics at global threshold.
    print(f"\n{'=' * 70}")
    print(f"MULTI-LABEL METRICS (threshold={THRESHOLD})")
    print(f"{'=' * 70}")

    preds = (all_probs >= THRESHOLD).astype(int)

    # Handle empty predictions: assign top-1.
    empty_mask = preds.sum(axis=1) == 0
    empty_count_before_fallback = int(empty_mask.sum())
    empty_rate_before_fallback = float(empty_mask.mean())

    if empty_mask.any():
        top1_indices = all_probs[empty_mask].argmax(axis=1)
        for row_idx, col_idx in zip(np.where(empty_mask)[0], top1_indices):
            preds[row_idx, col_idx] = 1

    avg_labels_pred = preds.sum(axis=1).mean()
    avg_labels_true = all_targets.sum(axis=1).mean()

    macro_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_targets, preds, average="micro", zero_division=0)
    weighted_f1 = f1_score(all_targets, preds, average="weighted", zero_division=0)
    macro_p = precision_score(all_targets, preds, average="macro", zero_division=0)
    macro_r = recall_score(all_targets, preds, average="macro", zero_division=0)
    h_loss = hamming_loss(all_targets, preds)

    print(f"  Macro F1:      {macro_f1:.4f}")
    print(f"  Micro F1:      {micro_f1:.4f}")
    print(f"  Weighted F1:   {weighted_f1:.4f}")
    print(f"  Macro Prec:    {macro_p:.4f}")
    print(f"  Macro Recall:  {macro_r:.4f}")
    print(f"  Hamming Loss:  {h_loss:.4f}")
    print(f"  Avg labels predicted: {avg_labels_pred:.2f} (true: {avg_labels_true:.2f})")
    print(
        f"  Empty predictions before fallback: "
        f"{empty_count_before_fallback} ({empty_rate_before_fallback:.1%})"
    )

    # Top-1 accuracy for V1 comparison.
    print(f"\n{'=' * 70}")
    print("TOP-1 ACCURACY (direct comparison to V1)")
    print(f"{'=' * 70}")

    top1_pred_idx = all_probs.argmax(axis=1)
    true_first_cat_idx = test_df["first_cat_idx"].values

    top1_acc = accuracy_score(true_first_cat_idx, top1_pred_idx)
    top1_f1_macro = f1_score(true_first_cat_idx, top1_pred_idx, average="macro", zero_division=0)
    top1_f1_weighted = f1_score(true_first_cat_idx, top1_pred_idx, average="weighted", zero_division=0)

    print(f"  V3 top-1 accuracy:      {top1_acc:.4f}  ({top1_acc:.1%})")
    print(f"  V3 top-1 F1 macro:      {top1_f1_macro:.4f}")
    print(f"  V3 top-1 F1 weighted:   {top1_f1_weighted:.4f}")

    diff_acc = top1_acc - 0.735
    print(f"  Delta accuracy:         {diff_acc:+.4f}  ({diff_acc:+.1%})")

    # Per-class F1.
    print(f"\n{'=' * 70}")
    print(f"PER-CLASS F1 (multi-label, threshold={THRESHOLD})")
    print(f"{'=' * 70}")

    per_class_f1 = f1_score(all_targets, preds, average=None, zero_division=0)
    per_class_p = precision_score(all_targets, preds, average=None, zero_division=0)
    per_class_r = recall_score(all_targets, preds, average=None, zero_division=0)
    support = all_targets.sum(axis=0).astype(int)

    class_df = pd.DataFrame(
        {
            "category": categories,
            "f1": per_class_f1,
            "precision": per_class_p,
            "recall": per_class_r,
            "support": support,
        }
    ).sort_values("f1", ascending=False)

    print(f"{'Category':<10} {'F1':>6} {'Prec':>6} {'Recall':>6} {'Support':>8}")
    print("-" * 42)

    for _, row in class_df.iterrows():
        print(
            f"{row['category']:<10} "
            f"{row['f1']:>6.3f} "
            f"{row['precision']:>6.3f} "
            f"{row['recall']:>6.3f} "
            f"{int(row['support']):>8,}"
        )

    print("\nBottom 5 classes by F1:")
    for _, row in class_df.tail(5).iterrows():
        print(f"  {row['category']}: F1={row['f1']:.3f}, support={int(row['support']):,}")

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_df.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)

    summary = {
        "checkpoint": str(CHECKPOINT_DIR),
        "test_size": len(test_df),
        "threshold": THRESHOLD,
        "multi_label": {
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "hamming_loss": round(h_loss, 4),
            "avg_labels_predicted": round(avg_labels_pred, 2),
            "avg_labels_true": round(avg_labels_true, 2),
            "empty_before_fallback": empty_count_before_fallback,
            "empty_rate_before_fallback": round(empty_rate_before_fallback, 4),
        },
        "top1_vs_v1": {
            "v3_accuracy": round(top1_acc, 4),
            "v3_f1_macro": round(top1_f1_macro, 4),
            "v3_f1_weighted": round(top1_f1_weighted, 4),
            "v1_accuracy": 0.735,
            "v1_f1_macro": 0.61,
            "v1_f1_weighted": 0.73,
            "delta_accuracy": round(diff_acc, 4),
        },
    }

    write_json_atomic(OUTPUT_DIR / "test_summary.json", summary)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("  - per_class_metrics.csv")
    print("  - test_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
