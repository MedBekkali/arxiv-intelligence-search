"""
V3 multi-label classifier using fine-tuned SPECTER2 + LoRA.

Usage:
    from arxiv_intel.classifier import LoRAClassifier
    clf = LoRAClassifier("models/v3_lora_classifier_r32_alpha64_len384_batch64_expI/best_micro")
    results = clf.predict("Attention Is All You Need", "We propose a new simple network architecture...")
    # results = [("cs.CL", 0.97), ("cs.LG", 0.94), ...]
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


class Specter2LoRAClassifier(nn.Module):
    """Same architecture as training script — must match exactly."""

    def __init__(self, base_model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS pooling
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# Default per-class thresholds (optimized on validation set).
# Loaded from per_class_thresholds.json at runtime when available.
_DEFAULT_THRESHOLD = 0.90

_DEFAULT_PER_CLASS_THRESHOLDS = {
    "cs.AI": 0.58, "cs.AR": 0.95, "cs.CC": 0.93, "cs.CE": 0.94,
    "cs.CG": 0.95, "cs.CL": 0.87, "cs.CR": 0.94, "cs.CV": 0.77,
    "cs.CY": 0.94, "cs.DB": 0.95, "cs.DC": 0.92, "cs.DL": 0.95,
    "cs.DM": 0.94, "cs.DS": 0.94, "cs.ET": 0.94, "cs.FL": 0.95,
    "cs.GR": 0.94, "cs.GT": 0.95, "cs.HC": 0.95, "cs.IR": 0.95,
    "cs.IT": 0.90, "cs.LG": 0.61, "cs.LO": 0.95, "cs.MA": 0.95,
    "cs.MM": 0.87, "cs.MS": 0.91, "cs.NA": 0.95, "cs.NE": 0.94,
    "cs.NI": 0.94, "cs.OH": 0.84, "cs.OS": 0.94, "cs.PF": 0.92,
    "cs.PL": 0.95, "cs.RO": 0.90, "cs.SC": 0.91, "cs.SD": 0.95,
    "cs.SE": 0.95, "cs.SI": 0.95, "cs.SY": 0.95,
}


class LoRAClassifier:
    """
    High-level wrapper for loading a checkpoint and running inference.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Path to a checkpoint folder (e.g. best_micro/) containing:
        - adapter/          (LoRA weights)
        - tokenizer/        (tokenizer files)
        - classifier.pt     (classifier head)
        - categories.json   (label list)
        - training_config.json (hyperparams)
    device : str
        "cuda" or "cpu". Defaults to cuda if available.
    threshold : float or None
        Global sigmoid threshold (fallback if per-class thresholds unavailable).
        Ignored when per-class thresholds are loaded.
    per_class_thresholds_path : str or Path or None
        Path to per_class_thresholds.json. If None, searches:
        1. checkpoint_dir / ../../test_results/per_class_thresholds.json
        2. Falls back to built-in defaults.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | None = None,
        threshold: float = _DEFAULT_THRESHOLD,
        per_class_thresholds_path: str | Path | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.global_threshold = threshold

        # Load config.
        config_path = self.checkpoint_dir / "training_config.json"
        with open(config_path) as f:
            self.config = json.load(f)

        # Load categories.
        cats_path = self.checkpoint_dir / "categories.json"
        with open(cats_path) as f:
            self.categories = json.load(f)
        self.num_labels = len(self.categories)

        # Load per-class thresholds.
        self.per_class_thresholds = self._load_per_class_thresholds(
            per_class_thresholds_path
        )

        # Load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_dir / "tokenizer"
        )
        self.max_length = self.config.get("max_length", 384)

        # Build model: base encoder + LoRA + classifier head.
        base_model_name = self.config.get("model_name", "allenai/specter2_base")
        self.model = self._load_model(base_model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_per_class_thresholds(
        self, explicit_path: str | Path | None
    ) -> dict[str, float]:
        """Load per-class thresholds from JSON, with fallback chain."""
        # 1. Explicit path provided by caller.
        if explicit_path is not None:
            path = Path(explicit_path)
            if path.exists():
                with open(path) as f:
                    thresholds = json.load(f)
                print(f"[classifier] Loaded per-class thresholds from {path}")
                return thresholds

        # 2. Auto-discover relative to checkpoint dir.
        auto_path = (
            self.checkpoint_dir / ".." / ".." / "test_results" / "per_class_thresholds.json"
        ).resolve()
        if auto_path.exists():
            with open(auto_path) as f:
                thresholds = json.load(f)
            print(f"[classifier] Loaded per-class thresholds from {auto_path}")
            return thresholds

        # 3. Built-in defaults.
        print("[classifier] Using built-in per-class thresholds")
        return dict(_DEFAULT_PER_CLASS_THRESHOLDS)

    def _load_model(self, base_model_name: str) -> Specter2LoRAClassifier:
        # 1. Create the full model with base encoder.
        model = Specter2LoRAClassifier(
            base_model_name=base_model_name,
            num_labels=self.num_labels,
            dropout=0.0,  # no dropout at inference
        )

        # 2. Apply LoRA adapters.
        model.encoder = PeftModel.from_pretrained(
            model.encoder,
            self.checkpoint_dir / "adapter",
        )
        # Merge LoRA weights into base for faster inference.
        model.encoder = model.encoder.merge_and_unload()

        # 3. Load classifier head.
        classifier_state = torch.load(
            self.checkpoint_dir / "classifier.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.classifier.load_state_dict(classifier_state)

        return model

    def _get_threshold(self, category: str) -> float:
        """Return per-class threshold, falling back to global."""
        return self.per_class_thresholds.get(category, self.global_threshold)

    @torch.no_grad()
    def predict(
        self,
        title: str,
        abstract: str,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """
        Predict CS categories for a single paper.

        Returns list of (category, probability) tuples, sorted by probability desc.
        Only categories above their per-class threshold are returned.
        If no category passes its threshold, returns the single highest-scoring one.

        Parameters
        ----------
        threshold : float or None
            If provided, overrides per-class thresholds with a single global value.
        """
        use_global = threshold is not None

        # Tokenize.
        text = f"{title} [SEP] {abstract}"
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward.
        logits = self.model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Apply thresholds.
        results = []
        for i, p in enumerate(probs):
            cat = self.categories[i]
            t = threshold if use_global else self._get_threshold(cat)
            if p >= t:
                results.append((cat, float(p)))

        # Fallback: if nothing passes threshold, return top-1.
        if not results:
            top_idx = probs.argmax()
            results.append((self.categories[top_idx], float(probs[top_idx])))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @torch.no_grad()
    def predict_all_probs(
        self, title: str, abstract: str
    ) -> list[tuple[str, float]]:
        """Return ALL 39 categories with their probabilities (for visualization)."""
        text = f"{title} [SEP] {abstract}"
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        results = [(self.categories[i], float(p)) for i, p in enumerate(probs)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
