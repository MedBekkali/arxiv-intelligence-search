"""
train_lora_classifier.py — LoRA fine-tuning for SPECTER2 multi-label arXiv CS classification.

Checkpointed version.

Adds:
    - Separate --eval-batch-size so batch 64 training does not force huge eval batches.
    - Saves snapshot adapters at every evaluation step.
    - Saves separate best_macro/, best_micro/, best_weighted/ checkpoints.
    - Keeps backward-compatible best/ as an alias of best_macro/.
    - Saves rolling checkpoint_last.pt with trainable weights + optimizer/scheduler/scaler state.
    - Optional --resume-from checkpoint_last.pt.
    - Saves threshold sweep CSV at every evaluation.

Important:
    checkpoint_last.pt restores model/optimizer/scheduler/scaler state and counters.
    DataLoader mid-epoch order is not perfectly reproduced, but this is still enough
    to avoid losing the learned weights and optimizer state after an interruption.

Experiment E — changes from Experiment D:
    - pos_weight clamped to [1, max_pos_weight] (default 50) to prevent rare-class
      gradient domination that caused probability miscalibration in Exp D.
    - Threshold sweep lowered to 0.20-0.60 (was 0.70-0.99).
    - Default target_modules now includes "key" alongside "query" and "value".
    - Default LR lowered to 1e-4 (was 2e-4).
    - Default epochs raised to 2 (was 1).
    - Default eval_every set to 2000 (was 3000).

Experiment E run:
    python scripts/train_lora_classifier.py `
      --output-dir models/v3_lora_classifier_r32_alpha64_len384_batch64_expE `
      --epochs 2 `
      --batch-size 64 `
      --eval-batch-size 32 `
      --grad-accum-steps 1 `
      --max-length 384 `
      --lr 1e-4 `
      --lora-r 32 `
      --lora-alpha 64 `
      --target-modules query key value `
      --eval-every 2000 `
      --max-val-samples 30000 `
      --patience 5 `
      --thresholds "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60" `
      --fp16

Resume:
    python scripts/train_lora_classifier.py `
      --resume-from models/v3_lora_classifier_r32_alpha64_len384_batch64_expE/checkpoint_last.pt `
      ...same args as original run...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# ---------- Paths ----------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
CATEGORIES_PATH = PROJECT_ROOT / "models" / "v3_categories.json"
POS_WEIGHTS_PATH = PROJECT_ROOT / "models" / "v3_pos_weights.npy"
OUTPUT_DIR = PROJECT_ROOT / "models" / "v3_lora_classifier"


# ---------- Utilities ----------


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


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


def count_trainable_params(model: nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return trainable, total, pct


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def get_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only trainable weights for compact rolling resume checkpoint.

    The frozen SPECTER2 base is reloaded from Hugging Face/local cache on resume.
    This saves LoRA adapter weights + classifier head, not all 110M base params.
    """
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    full_state = model.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in full_state.items()
        if name in trainable_names
    }


# ---------- Dataset ----------


class ArxivMultiLabelDataset(Dataset):
    """Lazy tokenization dataset.

    The parquet stores sparse label indices instead of 39-dimensional multi-hot vectors.
    We create the multi-hot labels inside collate_fn batch-by-batch.
    """

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


class Specter2LoRAClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        target_modules: list[str],
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        classifier_dropout: float,
    ) -> None:
        super().__init__()

        base = AutoModel.from_pretrained(model_name)
        hidden_size = int(base.config.hidden_size)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_config)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_embedding))


# ---------- Metrics ----------


@torch.no_grad()
def collect_validation_outputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    loss_fn = nn.BCEWithLogitsLoss()

    for batch in tqdm(dataloader, desc="eval", leave=False):
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
        y_true = labels.detach().float().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(y_true)

        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int8)
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
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "avg_predicted_labels": float(preds.sum(axis=1).mean()),
        "empty_prediction_rate": float((preds.sum(axis=1) == 0).mean()),
    }


def threshold_sweep(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict[str, float]]:
    return [metrics_at_threshold(probs, labels, t) for t in thresholds]


def select_best(sweep: list[dict[str, float]], metric_name: str) -> dict[str, float]:
    return max(sweep, key=lambda d: d[metric_name])


def write_threshold_sweep(path: Path, rows: list[dict[str, float]], val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold",
        "val_loss",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
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


# ---------- Saving ----------


@dataclass
class SaveInfo:
    model_name: str
    num_labels: int
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    classifier_dropout: float
    target_modules: list[str]
    best_step: int
    best_epoch: float
    selected_metric: str
    selected_score: float
    best_threshold: float
    best_f1_macro: float
    best_f1_micro: float
    best_f1_weighted: float
    best_precision_macro: float
    best_recall_macro: float
    best_avg_predicted_labels: float
    best_empty_prediction_rate: float


def build_save_info(
    args: argparse.Namespace,
    num_labels: int,
    step: int,
    epoch_float: float,
    selected_metric: str,
    metrics: dict[str, float],
) -> SaveInfo:
    return SaveInfo(
        model_name=args.model_name,
        num_labels=num_labels,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        classifier_dropout=args.classifier_dropout,
        target_modules=args.target_modules,
        best_step=step,
        best_epoch=epoch_float,
        selected_metric=selected_metric,
        selected_score=float(metrics[selected_metric]),
        best_threshold=float(metrics["threshold"]),
        best_f1_macro=float(metrics["f1_macro"]),
        best_f1_micro=float(metrics["f1_micro"]),
        best_f1_weighted=float(metrics["f1_weighted"]),
        best_precision_macro=float(metrics["precision_macro"]),
        best_recall_macro=float(metrics["recall_macro"]),
        best_avg_predicted_labels=float(metrics["avg_predicted_labels"]),
        best_empty_prediction_rate=float(metrics["empty_prediction_rate"]),
    )


def save_model_artifact(
    model: Specter2LoRAClassifier,
    tokenizer: Any,
    categories: list[str],
    save_dir: Path,
    save_info: SaveInfo,
    overwrite: bool = True,
) -> None:
    if overwrite:
        safe_rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = save_dir / "adapter"
    tokenizer_dir = save_dir / "tokenizer"

    model.encoder.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save(model.classifier.state_dict(), save_dir / "classifier.pt")

    with open(save_dir / "categories.json", "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)

    with open(save_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(save_info), f, indent=2)


def save_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    optimizer_step: int,
    train_iter_count: int,
    best_scores: dict[str, float],
    best_steps: dict[str, int],
    evals_without_improvement: int,
    recent_losses: list[float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "trainable_model_state_dict": get_trainable_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "optimizer_step": optimizer_step,
        "train_iter_count": train_iter_count,
        "best_scores": best_scores,
        "best_steps": best_steps,
        "evals_without_improvement": evals_without_improvement,
        "recent_losses": recent_losses[-100:],
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, checkpoint_path)


def load_training_state(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["trainable_model_state_dict"], strict=False)
    if unexpected:
        log(f"Warning: unexpected keys when loading resume checkpoint: {unexpected}")
    # Missing frozen base keys are expected because we save only trainable weights.
    log(f"Loaded trainable model state from {checkpoint_path}")

    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    scaler.load_state_dict(state["scaler_state_dict"])

    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if device.type == "cuda" and state.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_random_state_all"])

    return state


# ---------- Training helpers ----------


def make_val_subset(dataset: Dataset, max_val_samples: int, seed: int) -> Dataset:
    if max_val_samples <= 0 or max_val_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=max_val_samples, replace=False)
    return Subset(dataset, idx.tolist())


def parse_thresholds(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    for v in values:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Threshold must be between 0 and 1: {v}")
    return values


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SPECTER2 with LoRA for arXiv CS multi-label classification.")

    parser.add_argument("--model-name", type=str, default="allenai/specter2_base")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help=(
            "Optional hard stop at this optimizer step. "
            "0 means unchanged behavior. Useful for short proxy experiments."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per GPU step.")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Validation batch size. Default: max(batch_size, 16).")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["query", "key", "value"],
        help="Module name suffixes to LoRA-adapt. For BERT/SciBERT, query key value is a safe start.",
    )

    parser.add_argument("--eval-every", type=int, default=2000, help="Evaluate every N optimizer steps.")
    parser.add_argument("--max-val-samples", type=int, default=30000, help="0 means full validation set.")
    parser.add_argument("--patience", type=int, default=5, help="Early stop after N evals without macro-F1 improvement.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated global thresholds to test on validation.",
    )
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=50.0,
        help="Clamp pos_weight to [1, max_pos_weight]. Prevents rare-class gradient domination. "
             "Set to 0 to disable clamping (use raw pos_weights).",
    )

    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows for maximum safety.")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA fp16 mixed precision.")
    parser.add_argument("--dry-run", action="store_true", help="Run one train batch and one eval batch, then exit.")

    parser.add_argument("--resume-from", type=Path, default=None, help="Path to checkpoint_last.pt to resume training state.")
    parser.add_argument("--save-snapshots", action="store_true", default=True, help="Save snapshots/step_XXXX at every eval.")
    parser.add_argument("--no-save-snapshots", dest="save_snapshots", action="store_false", help="Disable eval snapshots.")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_last.pt", help="Rolling resume checkpoint filename.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    thresholds = parse_thresholds(args.thresholds)

    log("Starting LoRA fine-tuning")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(cuda_memory_summary())

    # Metadata.
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories = json.load(f)
    num_labels = len(categories)
    pos_weights_np = np.load(POS_WEIGHTS_PATH).astype(np.float32)
    if pos_weights_np.shape != (num_labels,):
        raise ValueError(f"Expected pos_weights shape ({num_labels},), got {pos_weights_np.shape}")

    log(f"Loaded {num_labels} categories")
    log(f"Raw pos_weight range: min={pos_weights_np.min():.1f}, max={pos_weights_np.max():.1f}, median={np.median(pos_weights_np):.1f}")

    # --- CLAMP POS_WEIGHTS ---
    if args.max_pos_weight > 0:
        n_clamped = int((pos_weights_np > args.max_pos_weight).sum())
        pos_weights_np = np.clip(pos_weights_np, 1.0, args.max_pos_weight)
        log(f"Clamped pos_weight to [{1.0}, {args.max_pos_weight}] — {n_clamped} classes were capped")
        log(f"Clamped pos_weight range: min={pos_weights_np.min():.1f}, max={pos_weights_np.max():.1f}, median={np.median(pos_weights_np):.1f}")
    else:
        log("pos_weight clamping DISABLED (--max-pos-weight 0)")

    # Tokenizer + data.
    log(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log("Loading train/val parquet splits")
    train_dataset = ArxivMultiLabelDataset(TRAIN_PATH)
    val_dataset_full = ArxivMultiLabelDataset(VAL_PATH)
    val_dataset = make_val_subset(val_dataset_full, args.max_val_samples, args.seed)

    log(f"Train papers: {len(train_dataset):,}")
    log(f"Val papers used per eval: {len(val_dataset):,} / {len(val_dataset_full):,}")

    collator = PaperCollator(tokenizer=tokenizer, n_classes=num_labels, max_length=args.max_length)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    eval_batch_size = args.eval_batch_size or max(args.batch_size, 16)
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    # Model.
    log(f"Loading base model + LoRA adapters: {args.model_name}")
    try:
        model = Specter2LoRAClassifier(
            model_name=args.model_name,
            num_labels=num_labels,
            target_modules=args.target_modules,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            classifier_dropout=args.classifier_dropout,
        )
    except ValueError as e:
        log("PEFT could not attach LoRA to the requested target modules.")
        log("For BERT-like models, common target modules are: query value key dense")
        raise e

    model.to(device)
    trainable, total, pct = count_trainable_params(model)
    log(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    if hasattr(model.encoder, "print_trainable_parameters"):
        model.encoder.print_trainable_parameters()

    # Loss / optimizer / schedule.
    pos_weight = torch.tensor(pos_weights_np, dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_optimizer_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))

    # Optional hard stop for short proxy experiments.
    # This does not change the LR schedule; it only stops training earlier.
    if args.max_optimizer_steps > 0:
        stop_optimizer_steps = min(total_optimizer_steps, args.max_optimizer_steps)
    else:
        stop_optimizer_steps = total_optimizer_steps

    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.fp16 and device.type == "cuda",
    )

    log(f"Micro batch size: {args.batch_size}")
    log(f"Eval batch size: {eval_batch_size}")
    log(f"Gradient accumulation steps: {args.grad_accum_steps}")
    log(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    log(f"Optimizer steps planned by epochs: {total_optimizer_steps:,}")
    if args.max_optimizer_steps > 0:
        log(f"Hard stop at optimizer step: {stop_optimizer_steps:,}")
    log(f"Warmup steps: {warmup_steps:,}")
    log(f"Eval every optimizer steps: {args.eval_every:,}")

    # Output structure.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "metrics_history.csv"
    checkpoint_last_path = args.output_dir / args.checkpoint_name
    snapshots_dir = args.output_dir / "snapshots"
    sweeps_dir = args.output_dir / "threshold_sweeps"
    best_macro_dir = args.output_dir / "best_macro"
    best_micro_dir = args.output_dir / "best_micro"
    best_weighted_dir = args.output_dir / "best_weighted"
    best_dir = args.output_dir / "best"  # backward-compatible alias of best_macro

    history_fields = [
        "epoch",
        "step",
        "train_loss_recent",
        "val_loss",
        "macro_threshold",
        "macro_f1_macro",
        "macro_f1_micro",
        "macro_f1_weighted",
        "macro_precision_macro",
        "macro_recall_macro",
        "macro_avg_predicted_labels",
        "macro_empty_prediction_rate",
        "micro_threshold",
        "best_f1_micro",
        "weighted_threshold",
        "best_f1_weighted",
        "elapsed",
    ]
    if not args.resume_from or not history_path.exists():
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writeheader()

    # State variables.
    best_scores = {
        "f1_macro": -1.0,
        "f1_micro": -1.0,
        "f1_weighted": -1.0,
    }
    best_steps = {
        "f1_macro": 0,
        "f1_micro": 0,
        "f1_weighted": 0,
    }

    evals_without_improvement = 0
    recent_losses: list[float] = []
    optimizer_step = 0
    train_iter_count = 0
    last_eval_step: int | None = None

    # Optional resume.
    if args.resume_from is not None:
        log(f"Resuming from {args.resume_from}")
        resume_state = load_training_state(args.resume_from, model, optimizer, scheduler, scaler, device)
        optimizer_step = int(resume_state.get("optimizer_step", 0))
        train_iter_count = int(resume_state.get("train_iter_count", 0))
        best_scores = dict(resume_state.get("best_scores", best_scores))
        best_steps = dict(resume_state.get("best_steps", best_steps))
        evals_without_improvement = int(resume_state.get("evals_without_improvement", 0))
        recent_losses = list(resume_state.get("recent_losses", []))
        log(f"Resume counters: optimizer_step={optimizer_step:,}, train_iter_count={train_iter_count:,}")
        log(f"Resume best scores: {best_scores}")
        log("Note: training state is restored, but mid-epoch DataLoader order is only approximately resumed.")

    def save_eval_artifacts(
        step: int,
        epoch_float: float,
        sweep_rows: list[dict[str, float]],
        val_loss: float,
        macro_metrics: dict[str, float],
        micro_metrics: dict[str, float],
        weighted_metrics: dict[str, float],
    ) -> None:
        # Save threshold sweep for this eval.
        write_threshold_sweep(sweeps_dir / f"step_{step:06d}.csv", sweep_rows, val_loss)

        # Save an immutable snapshot every eval.
        if args.save_snapshots:
            snapshot_info = build_save_info(
                args=args,
                num_labels=num_labels,
                step=step,
                epoch_float=epoch_float,
                selected_metric="f1_macro",
                metrics=macro_metrics,
            )
            save_model_artifact(
                model,
                tokenizer,
                categories,
                snapshots_dir / f"step_{step:06d}",
                snapshot_info,
                overwrite=True,
            )

        # Save metric-specific bests.
        metric_to_dir = {
            "f1_macro": best_macro_dir,
            "f1_micro": best_micro_dir,
            "f1_weighted": best_weighted_dir,
        }
        metric_to_metrics = {
            "f1_macro": macro_metrics,
            "f1_micro": micro_metrics,
            "f1_weighted": weighted_metrics,
        }

        for metric_name, save_dir in metric_to_dir.items():
            selected_metrics = metric_to_metrics[metric_name]
            if selected_metrics[metric_name] > best_scores[metric_name]:
                best_scores[metric_name] = float(selected_metrics[metric_name])
                best_steps[metric_name] = step
                save_info = build_save_info(
                    args=args,
                    num_labels=num_labels,
                    step=step,
                    epoch_float=epoch_float,
                    selected_metric=metric_name,
                    metrics=selected_metrics,
                )
                save_model_artifact(model, tokenizer, categories, save_dir, save_info, overwrite=True)
                log(f"✓ New best {metric_name}: {best_scores[metric_name]:.4f} — saved to {save_dir}")

                # Backward compatibility: best/ mirrors best_macro/.
                if metric_name == "f1_macro":
                    save_model_artifact(model, tokenizer, categories, best_dir, save_info, overwrite=True)
                    log(f"  best/ alias updated → {best_dir}")

    def run_eval_and_maybe_save(step: int, epoch_float: float, train_loss_recent: float, start_time: float) -> bool:
        nonlocal evals_without_improvement, optimizer_step, train_iter_count, recent_losses, last_eval_step

        log(f"Evaluating at step {step:,} | epoch {epoch_float:.3f}")
        last_eval_step = step
        probs, labels, val_loss = collect_validation_outputs(model, val_loader, device, args.fp16)

        sweep_rows = threshold_sweep(probs, labels, thresholds)
        macro_metrics = select_best(sweep_rows, "f1_macro")
        micro_metrics = select_best(sweep_rows, "f1_micro")
        weighted_metrics = select_best(sweep_rows, "f1_weighted")

        row = {
            "epoch": f"{epoch_float:.6f}",
            "step": step,
            "train_loss_recent": f"{train_loss_recent:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "macro_threshold": f"{macro_metrics['threshold']:.2f}",
            "macro_f1_macro": f"{macro_metrics['f1_macro']:.6f}",
            "macro_f1_micro": f"{macro_metrics['f1_micro']:.6f}",
            "macro_f1_weighted": f"{macro_metrics['f1_weighted']:.6f}",
            "macro_precision_macro": f"{macro_metrics['precision_macro']:.6f}",
            "macro_recall_macro": f"{macro_metrics['recall_macro']:.6f}",
            "macro_avg_predicted_labels": f"{macro_metrics['avg_predicted_labels']:.4f}",
            "macro_empty_prediction_rate": f"{macro_metrics['empty_prediction_rate']:.6f}",
            "micro_threshold": f"{micro_metrics['threshold']:.2f}",
            "best_f1_micro": f"{micro_metrics['f1_micro']:.6f}",
            "weighted_threshold": f"{weighted_metrics['threshold']:.2f}",
            "best_f1_weighted": f"{weighted_metrics['f1_weighted']:.6f}",
            "elapsed": format_seconds(time.time() - start_time),
        }
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writerow(row)

        log(
            "VAL best_macro "
            f"loss={val_loss:.4f} "
            f"macroF1={macro_metrics['f1_macro']:.4f} "
            f"microF1={macro_metrics['f1_micro']:.4f} "
            f"weightedF1={macro_metrics['f1_weighted']:.4f} "
            f"threshold={macro_metrics['threshold']:.2f} "
            f"avg_labels={macro_metrics['avg_predicted_labels']:.2f} "
            f"empty={macro_metrics['empty_prediction_rate']:.3f}"
        )
        log(
            "VAL best_weighted "
            f"weightedF1={weighted_metrics['f1_weighted']:.4f} "
            f"macroF1={weighted_metrics['f1_macro']:.4f} "
            f"threshold={weighted_metrics['threshold']:.2f} "
            f"avg_labels={weighted_metrics['avg_predicted_labels']:.2f} "
            f"empty={weighted_metrics['empty_prediction_rate']:.3f}"
        )
        log(
            "VAL best_micro "
            f"microF1={micro_metrics['f1_micro']:.4f} "
            f"macroF1={micro_metrics['f1_macro']:.4f} "
            f"threshold={micro_metrics['threshold']:.2f} "
            f"avg_labels={micro_metrics['avg_predicted_labels']:.2f} "
            f"empty={micro_metrics['empty_prediction_rate']:.3f}"
        )

        old_best_macro = best_scores["f1_macro"]
        save_eval_artifacts(step, epoch_float, sweep_rows, val_loss, macro_metrics, micro_metrics, weighted_metrics)

        if macro_metrics["f1_macro"] > old_best_macro:
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1
            log(f"No macro-F1 improvement. Patience: {evals_without_improvement}/{args.patience}")

        save_training_state(
            checkpoint_last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            optimizer_step=optimizer_step,
            train_iter_count=train_iter_count,
            best_scores=best_scores,
            best_steps=best_steps,
            evals_without_improvement=evals_without_improvement,
            recent_losses=recent_losses,
        )
        log(f"✓ Rolling resume checkpoint saved → {checkpoint_last_path}")

        if device.type == "cuda":
            log(cuda_memory_summary())

        model.train()
        return evals_without_improvement >= args.patience

    # Training loop.
    start_time = time.time()
    stop_training = False

    model.train()
    log("Training begins")

    # Initial eval before any training gives a useful baseline.
    if not args.dry_run and args.resume_from is None:
        _ = run_eval_and_maybe_save(step=0, epoch_float=0.0, train_loss_recent=float("nan"), start_time=start_time)

    max_train_batches = int(math.ceil(len(train_loader) * args.epochs))
    pbar = tqdm(
        total=stop_optimizer_steps,
        initial=min(optimizer_step, stop_optimizer_steps),
        desc="optimizer steps",
    )
    optimizer.zero_grad(set_to_none=True)

    # If resumed, this approximate skip avoids immediately re-processing from train_iter_count=0.
    # It is not perfect mid-epoch reproducibility, but it is practical for a solo training script.
    skipped_batches = 0

    for epoch_idx in range(math.ceil(args.epochs)):
        for batch_idx, batch in enumerate(train_loader):
            if skipped_batches < train_iter_count:
                skipped_batches += 1
                continue

            progress_epoch = train_iter_count / max(len(train_loader), 1)
            if progress_epoch >= args.epochs:
                stop_training = True
                break

            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.fp16 and device.type == "cuda",
            ):
                logits = model(**batch)
                loss = loss_fn(logits, labels)
                loss_for_backward = loss / args.grad_accum_steps

            scaler.scale(loss_for_backward).backward()
            recent_losses.append(float(loss.item()))
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            should_step = (train_iter_count + 1) % args.grad_accum_steps == 0
            is_last_batch = (train_iter_count + 1) >= max_train_batches

            if should_step or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_step += 1
                pbar.update(1)

                recent_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
                pbar.set_postfix(
                    loss=f"{recent_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    best_macro=f"{best_scores['f1_macro']:.4f}",
                    best_weighted=f"{best_scores['f1_weighted']:.4f}",
                )

                if args.dry_run:
                    log("Dry run: completed one optimizer step.")
                    stop_training = True
                    break

                if args.eval_every > 0 and optimizer_step % args.eval_every == 0:
                    epoch_float = train_iter_count / max(len(train_loader), 1)
                    stop_training = run_eval_and_maybe_save(
                        step=optimizer_step,
                        epoch_float=epoch_float,
                        train_loss_recent=recent_loss,
                        start_time=start_time,
                    )
                    if stop_training:
                        log("Early stopping triggered.")
                        break

                if optimizer_step >= stop_optimizer_steps:
                    stop_training = True
                    break

            train_iter_count += 1

        if stop_training:
            break

    pbar.close()

    if args.dry_run:
        log("Running one eval batch for dry-run sanity check.")
        small_val = Subset(val_dataset, list(range(min(len(val_dataset), max(eval_batch_size, 8)))))
        small_val_loader = DataLoader(
            small_val,
            batch_size=max(min(eval_batch_size, args.batch_size), 8),
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        _ = collect_validation_outputs(model, small_val_loader, device, args.fp16)
        log("✓ Dry run passed.")
        return

    # Final eval.
    # Final eval. Skip if the last training step already triggered an eval.
    if last_eval_step != optimizer_step:
        recent_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
        final_epoch_float = min(args.epochs, train_iter_count / max(len(train_loader), 1))
        _ = run_eval_and_maybe_save(
            step=optimizer_step,
            epoch_float=final_epoch_float,
            train_loss_recent=recent_loss,
            start_time=start_time,
        )
    else:
        log(f"Skipping final eval because step {optimizer_step:,} was already evaluated.")

    log("─" * 70)
    log("Training complete")
    log(f"Elapsed: {format_seconds(time.time() - start_time)}")
    log(f"Best macro F1: {best_scores['f1_macro']:.4f} at step {best_steps['f1_macro']:,} → {best_macro_dir}")
    log(f"Best micro F1: {best_scores['f1_micro']:.4f} at step {best_steps['f1_micro']:,} → {best_micro_dir}")
    log(f"Best weighted F1: {best_scores['f1_weighted']:.4f} at step {best_steps['f1_weighted']:,} → {best_weighted_dir}")
    log(f"Backward-compatible best checkpoint: {best_dir}")
    log(f"Rolling resume checkpoint: {checkpoint_last_path}")
    log(f"Metrics history: {history_path}")
    log(f"Threshold sweeps: {sweeps_dir}")
    if args.save_snapshots:
        log(f"Eval snapshots: {snapshots_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()