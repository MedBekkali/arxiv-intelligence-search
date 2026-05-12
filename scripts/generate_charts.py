import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("models/v3_lora_classifier_r32_alpha64_len384_batch64_expI")
METRICS_CSV = MODEL_DIR / "metrics_history.csv"
TEST_RESULTS_DIR = MODEL_DIR / "test_results"
CHARTS_DIR = Path("charts")

# Style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.8,
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
})

# Colors
BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"
GRAY = "#8b949e"


def load_metrics():
    df = pd.read_csv(METRICS_CSV)
    return df

# ── Chart 1: Training Curves (Loss + F1) ─────────────────────────────────────

def chart_training_curves(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = df["step"]

    # Left: validation loss
    ax1.plot(steps, df["val_loss"], color=BLUE, linewidth=2, marker="o", markersize=6)
    ax1.set_xlabel("Optimizer Step")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # Right: F1 scores
    ax2.plot(steps, df["macro_f1_macro"], color=GREEN, linewidth=2, marker="o",
             markersize=6, label="Macro F1")
    ax2.plot(steps, df["best_f1_micro"], color=BLUE, linewidth=2, marker="s",
             markersize=6, label="Micro F1")
    ax2.plot(steps, df["best_f1_weighted"], color=ORANGE, linewidth=2, marker="^",
             markersize=6, label="Weighted F1")
    ax2.set_xlabel("Optimizer Step")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Scores", fontsize=13, fontweight="bold")
    ax2.legend(framealpha=0.3, edgecolor="#30363d")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    fig.suptitle("Experiment I — Training Progress (SPECTER2 + LoRA)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "training_curves.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ training_curves.png")
    plt.close()

# ── Chart 2: Experiment Comparison ────────────────────────────────────────────

def chart_experiment_comparison():
    experiments = {
        "D\nQV, cap=∞": 0.557,
        "E\nQKV, cap=50": 0.578,
        "F\nQKV, cap=75": 0.550,
        "G\nQKV, cap=100": 0.546,
        "H\nQKV, cap=150": 0.544,
        "I\nQKVD, cap=50": 0.600,
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(experiments.keys())
    values = list(experiments.values())
    colors = [GRAY, BLUE, GRAY, GRAY, GRAY, GREEN]

    bars = ax.bar(names, values, color=colors, width=0.6, edgecolor="#30363d", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#c9d1d9")

    # Annotations
    ax.annotate("Winner", xy=(5, 0.600), xytext=(4.2, 0.615),
                fontsize=11, fontweight="bold", color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

    ax.annotate("Proxy runs\n(3000 steps each)", xy=(2, 0.535), xytext=(2, 0.510),
                fontsize=9, color=GRAY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    ax.set_ylabel("Best Macro F1")
    ax.set_title("LoRA Experiment Comparison — 6 Runs, 34h GPU Total",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.50, 0.63)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "experiment_comparison.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ experiment_comparison.png")
    plt.close()


# ── Chart 3: Ablation — pos_weight cap ────────────────────────────────────────

def chart_posweight_ablation():
    """Line chart showing macro F1 vs pos_weight cap (proxy runs)."""
    caps = [50, 75, 100, 150]
    f1s = [0.578, 0.550, 0.546, 0.544]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(caps, f1s, color=ORANGE, linewidth=2.5, marker="o", markersize=10, zorder=5)

    # Highlight cap=50
    ax.scatter([50], [0.578], color=GREEN, s=150, zorder=10, edgecolors="white", linewidth=2)
    ax.annotate("Cap=50 (best)", xy=(50, 0.578), xytext=(70, 0.585),
                fontsize=11, fontweight="bold", color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

    ax.set_xlabel("pos_weight Cap")
    ax.set_ylabel("Macro F1 (at 3000 steps)")
    ax.set_title("pos_weight Capping Ablation — Higher Caps Don't Help",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(caps)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "posweight_ablation.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ posweight_ablation.png")
    plt.close()


# ── Chart 4: LoRA Targets Ablation ───────────────────────────────────────────

def chart_lora_targets():
    """Bar chart: QV vs QKV vs QKVD."""
    targets = {
        "Q + V\n(1.2M params)": 0.557,
        "Q + K + V\n(1.8M params)": 0.578,
        "Q + K + V + Dense\n(5.4M params)": 0.600,
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(targets.keys())
    values = list(targets.values())
    colors = [GRAY, BLUE, GREEN]

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="#30363d")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#c9d1d9")

    ax.set_ylabel("Best Macro F1")
    ax.set_title("LoRA Target Modules — Adding Dense Is the Key",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.52, 0.63)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lora_targets_ablation.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ lora_targets_ablation.png")
    plt.close()


# ── Chart 5: V1 vs V3 Comparison ─────────────────────────────────────────────

def chart_v1_vs_v3(test_summary=None):
    """Side-by-side grouped bar chart: V1 vs V3 metrics."""
    metrics = ["Accuracy\n(top-1)", "F1 Macro", "F1 Weighted"]
    v1_vals = [0.735, 0.61, 0.73]

    if test_summary:
        t1 = test_summary["top1_vs_v1"]
        v3_vals = [t1["v3_accuracy"], t1["v3_f1_macro"], t1["v3_f1_weighted"]]
    else:
        # Placeholder from validation (will be replaced after Day 8)
        v3_vals = [None, 0.60, 0.668]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(metrics))
    width = 0.30

    bars1 = ax.bar(x - width/2, v1_vals, width, color=GRAY, label="V1 (TF-IDF + LogReg)",
                   edgecolor="#30363d")

    v3_display = [v if v is not None else 0 for v in v3_vals]
    bars2 = ax.bar(x + width/2, v3_display, width, color=GREEN,
                   label="V3 (SPECTER2 + LoRA)", edgecolor="#30363d")

    # Labels
    for bar, val in zip(bars1, v1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=GRAY)

    for bar, val in zip(bars2, v3_vals):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=GREEN)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                    "pending", ha="center", va="bottom", fontsize=9, color=GRAY,
                    fontstyle="italic")

    ax.set_ylabel("Score")
    title = "V1 vs V3 — Single-Label Comparison"
    if not test_summary:
        title += " (val set, test pending)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.85)
    ax.legend(framealpha=0.3, edgecolor="#30363d", loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Note about multi-label
    ax.text(0.02, 0.02, "Note: V3 also does multi-label (39 independent predictions).\n"
            "V1 is single-label only. Same F1 = harder task solved.",
            transform=ax.transAxes, fontsize=8, color=GRAY, verticalalignment="bottom")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "v1_vs_v3.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ v1_vs_v3.png")
    plt.close()


# ── Chart 6: Per-Class F1 (after Day 8) ──────────────────────────────────────

def chart_per_class_f1():
    """Horizontal bar chart of F1 per category, sorted."""
    csv_path = TEST_RESULTS_DIR / "per_class_metrics.csv"
    if not csv_path.exists():
        print("  ⏭ per_class_f1.png — skipped (run evaluate_test.py first)")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = []
    for f1 in df["f1"]:
        if f1 >= 0.7:
            colors.append(GREEN)
        elif f1 >= 0.5:
            colors.append(BLUE)
        elif f1 >= 0.3:
            colors.append(ORANGE)
        else:
            colors.append(RED)

    bars = ax.barh(df["category"], df["f1"], color=colors, height=0.7, edgecolor="#30363d",
                   linewidth=0.5)

    # Value labels
    for bar, f1 in zip(bars, df["f1"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f}", ha="left", va="center", fontsize=9, color="#c9d1d9")

    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 — Multi-Label, Threshold=0.90",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN, label="≥ 0.70 (strong)"),
        Patch(facecolor=BLUE, label="0.50–0.70 (decent)"),
        Patch(facecolor=ORANGE, label="0.30–0.50 (weak)"),
        Patch(facecolor=RED, label="< 0.30 (struggling)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.3, edgecolor="#30363d")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "per_class_f1.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ per_class_f1.png")
    plt.close()


# ── Chart 7: Architecture Diagram ────────────────────────────────────────────

def chart_architecture():
    """Simple pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, "902K\narXiv Papers", GRAY),
        (2.7, 1.5, "Data\nCleaning", GRAY),
        (4.9, 1.5, "V1: TF-IDF\n+ LogReg", BLUE),
        (7.1, 1.5, "SPECTER2\nEmbeddings", ORANGE),
        (9.3, 1.5, "FAISS\nHNSW Index", PURPLE),
        (11.5, 1.5, "Streamlit\nApp", GREEN),
    ]

    for x, y, text, color in boxes:
        rect = plt.Rectangle((x, y), 1.8, 1.2, facecolor=color + "22",
                              edgecolor=color, linewidth=2, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.9, y + 0.6, text, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#c9d1d9", zorder=3)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.8
        x2 = boxes[i + 1][0]
        y = 2.1
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.5))

    # Branch: LoRA
    ax.annotate("", xy=(9.3, 3.2), xytext=(8.0, 3.2),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    rect = plt.Rectangle((5.8, 2.9), 2.2, 0.7, facecolor=GREEN + "22",
                          edgecolor=GREEN, linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(6.9, 3.25, "LoRA\nFine-tuning", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#c9d1d9", zorder=3)

    # Branch: RAG
    rect2 = plt.Rectangle((9.3, 2.9), 2.2, 0.7, facecolor=CYAN + "22",
                           edgecolor=CYAN, linewidth=2, zorder=2)
    ax.add_patch(rect2)
    ax.text(10.4, 3.25, "RAG\n(Claude Haiku)", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#c9d1d9", zorder=3)

    ax.annotate("", xy=(11.5, 2.5), xytext=(11.5, 3.2),
                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.5))

    fig.suptitle("arXiv Intelligence Search — Architecture",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "architecture.png", bbox_inches="tight", pad_inches=0.3)
    print("  ✓ architecture.png")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-test", action="store_true",
                        help="Include test-set charts (requires evaluate_test.py results)")
    args = parser.parse_args()

    CHARTS_DIR.mkdir(exist_ok=True)

    print("Generating charts...\n")

    # Always generate these
    if METRICS_CSV.exists():
        df = load_metrics()
        chart_training_curves(df)
    else:
        print(f"  ⏭ training_curves.png — skipped ({METRICS_CSV} not found)")

    chart_experiment_comparison()
    chart_posweight_ablation()
    chart_lora_targets()
    chart_architecture()

    # Test-set charts
    test_summary = None
    summary_path = TEST_RESULTS_DIR / "test_summary.json"
    if args.with_test and summary_path.exists():
        with open(summary_path) as f:
            test_summary = json.load(f)
        chart_per_class_f1()

    chart_v1_vs_v3(test_summary)

    print(f"\nAll charts saved to {CHARTS_DIR}/")
    if not args.with_test:
        print("Tip: run with --with-test after evaluate_test.py for the full set.")


if __name__ == "__main__":
    main()
