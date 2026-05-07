"""
evaluate_lora_classifier.py — Evaluate saved SPECTER2 + LoRA multi-label classifier.

Purpose:
    Load the best checkpoint from models/v3_lora_classifier/best and test a wider
    threshold grid WITHOUT retraining.

Reads:
    models/v3_lora_classifier/best/adapter/
    models/v3_lora_classifier/best/classifier.pt
    models/v3_lora_classifier/best/tokenizer/
    models/v3_lora_classifier/best/categories.json
    models/v3_lora_classifier/best/training_config.json
    data/processed/val.parquet or data/processed/test.parquet

Writes:
    models/v3_lora_classifier/eval_threshold_sweep_<split>.csv
    models/v3_lora_classifier/eval_per_class_<split>.csv

Example:
    python scripts/evaluate_lora_classifier.py --split val --fp16

Full validation set:
    python scripts/evaluate_lora_classifier.py --split val --max-samples 0 --fp16

Test set, only after model selection is finished:
    python scripts/evaluate_lora_classifier.py --split test --max-samples 0 --fp16
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ---------- Paths ----------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "v3_lora_classifier" / "best"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "v3_lora_classifier"


# ---------- Utilities ----------


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_memory_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    name = torch.cuda.get_device_name(0)
    return (
        f"{name} | allocated={allocated:.2f}GB "
        f"reserved={reserved:.2f}GB max_allocated={max_allocated:.2f}GB"
    )


def parse_thresholds(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {value}")
    return values


# ---------- Dataset ----------


class ArxivMultiLabelDataset(Dataset):
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")

        df = pd.read_parquet(path, columns=["id", "title", "abstract", "label_indices"])
        self.ids = df["id"].astype(str).tolist()
        self.titles = df["title"].fillna("").astype(str).tolist()
        self.abstracts = df["abstract"].fillna("").astype(str).tolist()
        self.label_indices = df["label_indices"].tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "id": self.ids[idx],
            "title": self.titles[idx],
            "abstract": self.abstracts[idx],
            "label_indices": self.label_indices[idx],
        }


class PaperCollator:
    def __init__(self, tokenizer: Any, n_classes: int, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.n_classes = n_classes
        self.max_length = max_length
        self.sep = tokenizer.sep_token or "[SEP]"

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [f"{item['title']} {self.sep} {item['abstract']}" for item in batch]

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        labels = torch.zeros((len(batch), self.n_classes), dtype=torch.float32)
        for row, item in enumerate(batch):
            idxs = item["label_indices"]
            if isinstance(idxs, np.ndarray):
                idxs = idxs.tolist()
            if idxs is None:
                idxs = []
            for col in idxs:
                labels[row, int(col)] = 1.0

        encoded["labels"] = labels
        return encoded


# ---------- Model ----------


class LoadedLoRAClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int, classifier_path: Path) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_labels)

        state = torch.load(classifier_path, map_location="cpu")
        self.classifier.load_state_dict(state)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_embedding)


def load_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[LoadedLoRAClassifier, Any, list[str], dict[str, Any]]:
    adapter_dir = checkpoint_dir / "adapter"
    tokenizer_dir = checkpoint_dir / "tokenizer"
    classifier_path = checkpoint_dir / "classifier.pt"
    categories_path = checkpoint_dir / "categories.json"
    config_path = checkpoint_dir / "training_config.json"

    missing = [
        p for p in [adapter_dir, tokenizer_dir, classifier_path, categories_path, config_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoint files/directories:\n" + "\n".join(str(p) for p in missing))

    with open(categories_path, "r", encoding="utf-8") as f:
        categories = json.load(f)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config.get("model_name", "allenai/specter2_base")

    log(f"Loading tokenizer from {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    log(f"Loading base model: {model_name}")
    base = AutoModel.from_pretrained(model_name)
    hidden_size = int(base.config.hidden_size)

    log(f"Loading LoRA adapter from {adapter_dir}")
    encoder = PeftModel.from_pretrained(base, adapter_dir)

    model = LoadedLoRAClassifier(
        encoder=encoder,
        hidden_size=hidden_size,
        num_labels=len(categories),
        classifier_path=classifier_path,
    )
    model.to(device)
    model.eval()

    return model, tokenizer, categories, config


# ---------- Metrics ----------


@torch.no_grad()
def collect_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(dataloader, desc="evaluating", leave=True):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=fp16 and device.type == "cuda",
        ):
            logits = model(**batch)
            loss = loss_fn(logits, labels)

        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
        y_true = labels.detach().float().cpu().numpy().astype(np.int8)

        all_probs.append(probs)
        all_labels.append(y_true)

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    val_loss = total_loss / max(total_examples, 1)
    return probs_np, labels_np, val_loss


def metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(np.int8)

    return {
        "threshold": float(threshold),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "avg_predicted_labels": float(preds.sum(axis=1).mean()),
        "empty_prediction_rate": float((preds.sum(axis=1) == 0).mean()),
    }


def per_class_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    categories: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    preds = (probs >= threshold).astype(np.int8)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        zero_division=0,
    )

    predicted_support = preds.sum(axis=0)
    rows: list[dict[str, Any]] = []
    for i, category in enumerate(categories):
        rows.append(
            {
                "category": category,
                "true_support": int(support[i]),
                "predicted_support": int(predicted_support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        )
    return rows


def write_threshold_sweep(path: Path, rows: list[dict[str, Any]], val_loss: float) -> None:
    fields = [
        "threshold",
        "val_loss",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "precision_micro",
        "recall_macro",
        "recall_micro",
        "avg_predicted_labels",
        "empty_prediction_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["val_loss"] = val_loss
            writer.writerow(out)


def write_per_class(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["category", "true_support", "predicted_support", "precision", "recall", "f1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved SPECTER2 + LoRA classifier with threshold sweep.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=30000, help="0 means full split.")
    parser.add_argument("--max-length", type=int, default=None, help="Override checkpoint max_length if provided.")
    parser.add_argument("--thresholds", type=str, default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    thresholds = parse_thresholds(args.thresholds)
    device = get_device()

    log("Starting LoRA classifier evaluation")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(cuda_memory_summary())

    model, tokenizer, categories, config = load_checkpoint(args.checkpoint_dir, device)

    max_length = args.max_length if args.max_length is not None else int(config.get("max_length", 256))
    log(f"Using max_length={max_length}")
    log(f"Checkpoint best threshold was: {config.get('best_threshold', 'unknown')}")
    log(f"Checkpoint best macro F1 was: {config.get('best_f1_macro', 'unknown')}")

    split_path = PROCESSED_DIR / f"{args.split}.parquet"
    dataset_full = ArxivMultiLabelDataset(split_path)

    if args.max_samples > 0 and args.max_samples < len(dataset_full):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(dataset_full), size=args.max_samples, replace=False)
        dataset: Dataset = Subset(dataset_full, idx.tolist())
    else:
        dataset = dataset_full

    log(f"Evaluating split: {args.split}")
    log(f"Samples used: {len(dataset):,} / {len(dataset_full):,}")

    collator = PaperCollator(tokenizer=tokenizer, n_classes=len(categories), max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    probs, labels, val_loss = collect_outputs(model, dataloader, device, args.fp16)
    log(f"Unweighted BCE validation loss: {val_loss:.4f}")
    log(f"True avg labels per paper in evaluated sample: {labels.sum(axis=1).mean():.4f}")

    sweep_rows = [metrics_at_threshold(probs, labels, t) for t in thresholds]
    best = max(sweep_rows, key=lambda row: row["f1_macro"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.output_dir / f"eval_threshold_sweep_{args.split}.csv"
    per_class_path = args.output_dir / f"eval_per_class_{args.split}.csv"

    write_threshold_sweep(sweep_path, sweep_rows, val_loss)
    per_rows = per_class_metrics(probs, labels, categories, best["threshold"])
    write_per_class(per_class_path, per_rows)

    log("Threshold sweep:")
    for row in sweep_rows:
        marker = "<-- best" if row["threshold"] == best["threshold"] else ""
        log(
            f"  t={row['threshold']:.2f} | "
            f"macroF1={row['f1_macro']:.4f} | "
            f"microF1={row['f1_micro']:.4f} | "
            f"weightedF1={row['f1_weighted']:.4f} | "
            f"P_macro={row['precision_macro']:.4f} | "
            f"R_macro={row['recall_macro']:.4f} | "
            f"avg_labels={row['avg_predicted_labels']:.2f} | "
            f"empty={row['empty_prediction_rate']:.3f} {marker}"
        )

    worst_classes = sorted(per_rows, key=lambda r: r["f1"])[:8]
    best_classes = sorted(per_rows, key=lambda r: r["f1"], reverse=True)[:8]

    log("Worst 8 classes by F1 at best threshold:")
    for row in worst_classes:
        log(
            f"  {row['category']:8s} | "
            f"F1={row['f1']:.4f} | P={row['precision']:.4f} | R={row['recall']:.4f} | "
            f"true={row['true_support']:,} | pred={row['predicted_support']:,}"
        )

    log("Best 8 classes by F1 at best threshold:")
    for row in best_classes:
        log(
            f"  {row['category']:8s} | "
            f"F1={row['f1']:.4f} | P={row['precision']:.4f} | R={row['recall']:.4f} | "
            f"true={row['true_support']:,} | pred={row['predicted_support']:,}"
        )

    log("─" * 70)
    log("Evaluation complete")
    log(f"Best threshold by macro F1: {best['threshold']:.2f}")
    log(f"Best macro F1: {best['f1_macro']:.4f}")
    log(f"Best micro F1: {best['f1_micro']:.4f}")
    log(f"Best weighted F1: {best['f1_weighted']:.4f}")
    log(f"Avg predicted labels: {best['avg_predicted_labels']:.2f}")
    log(f"Threshold sweep saved: {sweep_path}")
    log(f"Per-class metrics saved: {per_class_path}")

    if device.type == "cuda":
        log(cuda_memory_summary())


if __name__ == "__main__":
    main()