"""
embed_papers.py — Compute SPECTER2 embeddings for the full arXiv CS corpus.

Reads:
    data/processed/arxiv_cs_clean.parquet   (~902k papers, V1 cleaning output)

Writes:
    models/v3_specter2_embeddings.npy       (~2.7 GB, shape: (n_papers, 768), float32)
    models/v3_papers_meta.parquet           (~50 MB, aligned metadata for each paper)

Behavior:
    - Runs a 1000-paper speed test before committing to the full run
    - Checkpoints intermediate results every 50,000 papers
    - Resumes from the last checkpoint if interrupted
    - Logs GPU temperature periodically (warns if throttling)

Usage:
    python scripts/embed_papers.py
    python scripts/embed_papers.py --resume       # explicit resume (default behavior)
    python scripts/embed_papers.py --restart      # ignore checkpoints, start over
    python scripts/embed_papers.py --batch-size 16  # smaller batches if OOM
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------- Configuration ----------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'arxiv_cs_clean.parquet'
MODELS_DIR = PROJECT_ROOT / 'models'
EMBEDDINGS_PATH = MODELS_DIR / 'v3_specter2_embeddings.npy'
META_PATH = MODELS_DIR / 'v3_papers_meta.parquet'
CHECKPOINT_PATH = MODELS_DIR / 'v3_specter2_embeddings.checkpoint.npy'
PROGRESS_PATH = MODELS_DIR / 'v3_embedding_progress.txt'

MODEL_NAME = 'allenai/specter2_base'
EMBEDDING_DIM = 768
META_COLUMNS = ['id', 'title', 'authors', 'year', 'first_cat', 'cs_cats']

# Defaults — overridable via CLI
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SEQ_LENGTH = 256       # SPECTER2's training length
CHECKPOINT_EVERY = 50_000          # save partial progress every N papers
GPU_CHECK_EVERY = 10_000           # check GPU temp every N papers
GPU_TEMP_WARN = 80                 # warn above this temp (Celsius)


# ---------- Helpers ----------

def get_gpu_temp() -> int | None:
    """Return current GPU temperature in Celsius, or None if unavailable."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split('\n')[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def log(msg: str) -> None:
    """Timestamped log line."""
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ---------- Main pipeline ----------

def run_speed_test(model: SentenceTransformer, abstracts: list[str], batch_size: int) -> float:
    """Embed a small sample to estimate full-run time. Returns papers/second."""
    log(f"Running speed test on {len(abstracts):,} papers...")
    t0 = time.time()
    _ = model.encode(
        abstracts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.time() - t0
    speed = len(abstracts) / elapsed
    log(f"Speed test complete: {speed:.0f} papers/second ({elapsed:.1f}s for {len(abstracts):,} papers)")
    return speed


def embed_corpus(
    model: SentenceTransformer,
    abstracts: list[str],
    batch_size: int,
    start_index: int = 0,
    existing_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Embed all abstracts in chunks, checkpointing periodically.

    If start_index > 0, resumes from existing_embeddings (must have shape (start_index, 768)).
    """
    n_total = len(abstracts)
    embeddings = np.zeros((n_total, EMBEDDING_DIM), dtype=np.float32)

    if existing_embeddings is not None and start_index > 0:
        embeddings[:start_index] = existing_embeddings
        log(f"Resuming from index {start_index:,} ({start_index / n_total * 100:.1f}% done)")

    pbar = tqdm(
        total=n_total,
        initial=start_index,
        unit='paper',
        unit_scale=True,
        desc='Embedding',
    )

    last_gpu_check = start_index
    last_checkpoint = start_index

    # Process in chunks of CHECKPOINT_EVERY for safe checkpointing
    chunk_size = CHECKPOINT_EVERY
    for chunk_start in range(start_index, n_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_total)
        chunk = abstracts[chunk_start:chunk_end]

        chunk_embeddings = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,    # tqdm above tracks at the chunk level
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings[chunk_start:chunk_end] = chunk_embeddings

        pbar.update(len(chunk))

        # GPU temperature check
        if chunk_end - last_gpu_check >= GPU_CHECK_EVERY:
            temp = get_gpu_temp()
            if temp is not None:
                pbar.set_postfix({'GPU_C': temp})
                if temp >= GPU_TEMP_WARN:
                    log(f"⚠️  GPU at {temp}°C — thermal throttling likely. Consider pausing.")
            last_gpu_check = chunk_end

        # Checkpoint
        if chunk_end - last_checkpoint >= CHECKPOINT_EVERY or chunk_end == n_total:
            log(f"Checkpoint: saving {chunk_end:,} embeddings to {CHECKPOINT_PATH.name}")
            np.save(CHECKPOINT_PATH, embeddings[:chunk_end])
            PROGRESS_PATH.write_text(str(chunk_end))
            last_checkpoint = chunk_end

    pbar.close()
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute SPECTER2 embeddings for arXiv CS corpus.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Encoding batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--max-seq-length', type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help=f'Max token length per abstract (default: {DEFAULT_MAX_SEQ_LENGTH})')
    parser.add_argument('--restart', action='store_true',
                        help='Ignore existing checkpoints and start from scratch')
    parser.add_argument('--skip-speed-test', action='store_true',
                        help='Skip the 1000-paper warmup speed test')
    args = parser.parse_args()

    overall_start = time.time()

    # ---------- 1. Verify environment ----------
    log("Checking environment...")
    if not torch.cuda.is_available():
        log("⚠️  No CUDA GPU detected. Embedding on CPU will be ~30x slower.")
        device = "cpu"
    else:
        device = "cuda"
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()

    if not DATA_PATH.exists():
        sys.exit(f"❌ Data file not found: {DATA_PATH}")

    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    # ---------- 2. Load corpus ----------
    log(f"Loading corpus from {DATA_PATH.name}...")
    t0 = time.time()
    df = pd.read_parquet(DATA_PATH)
    log(f"Loaded {len(df):,} papers in {time.time() - t0:.1f}s")

    # Verify required columns
    required = {'id', 'abstract', 'title'}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"❌ Missing required columns: {missing}")

    # Build the input text. SPECTER2 was trained on title + [SEP] + abstract.
    log("Building input texts (title + abstract)...")
    sep = model_sep_token = "[SEP]"
    df['_input'] = df['title'].fillna('') + ' ' + sep + ' ' + df['abstract'].fillna('')
    abstracts = df['_input'].tolist()

    # ---------- 3. Resume detection ----------
    start_index = 0
    existing_embeddings = None

    if not args.restart and CHECKPOINT_PATH.exists() and PROGRESS_PATH.exists():
        try:
            saved_index = int(PROGRESS_PATH.read_text().strip())
            existing_embeddings = np.load(CHECKPOINT_PATH)
            if existing_embeddings.shape[0] == saved_index and saved_index <= len(abstracts):
                start_index = saved_index
                log(f"Found checkpoint at {start_index:,} papers — resuming.")
            else:
                log("Checkpoint shape mismatch — starting fresh.")
                existing_embeddings = None
        except Exception as e:
            log(f"Could not load checkpoint ({e}) — starting fresh.")
            existing_embeddings = None

    if start_index >= len(abstracts):
        log("All papers already embedded. Loading checkpoint as final result.")
        all_embeddings = existing_embeddings
    else:
        # ---------- 4. Load model ----------
        log(f"Loading {MODEL_NAME}...")
        t0 = time.time()
        model = SentenceTransformer(MODEL_NAME, device=device)
        model.max_seq_length = args.max_seq_length
        log(f"Model loaded in {time.time() - t0:.1f}s (embedding dim: {EMBEDDING_DIM})")

        # ---------- 5. Speed test ----------
        if not args.skip_speed_test and start_index == 0:
            sample = abstracts[:1000]
            speed = run_speed_test(model, sample, args.batch_size)
            est_total_min = (len(abstracts) / speed) / 60
            log(f"Estimated full run time: {est_total_min:.0f} minutes for {len(abstracts):,} papers")
            log(f"Starting in 5 seconds... (Ctrl+C to abort)")
            time.sleep(5)

        # ---------- 6. Full embedding run ----------
        log(f"Beginning full embedding run with batch_size={args.batch_size}")
        all_embeddings = embed_corpus(
            model,
            abstracts,
            batch_size=args.batch_size,
            start_index=start_index,
            existing_embeddings=existing_embeddings,
        )

    # ---------- 7. Save final outputs ----------
    log(f"Saving final embeddings to {EMBEDDINGS_PATH.name} ({all_embeddings.nbytes / 1e9:.2f} GB)...")
    np.save(EMBEDDINGS_PATH, all_embeddings)

    log(f"Saving aligned metadata to {META_PATH.name}...")
    keep_cols = [c for c in META_COLUMNS if c in df.columns]
    df[keep_cols].to_parquet(META_PATH, index=False)

    # Clean up checkpoint files now that final outputs exist
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    # ---------- 8. Summary ----------
    overall_elapsed = time.time() - overall_start
    log("─" * 60)
    log(f"✓ Done in {format_duration(overall_elapsed)}")
    log(f"  Papers embedded: {all_embeddings.shape[0]:,}")
    log(f"  Embedding dim: {all_embeddings.shape[1]}")
    log(f"  Embeddings file: {EMBEDDINGS_PATH} ({EMBEDDINGS_PATH.stat().st_size / 1e9:.2f} GB)")
    log(f"  Metadata file:   {META_PATH} ({META_PATH.stat().st_size / 1e6:.1f} MB)")
    log(f"  Avg speed: {all_embeddings.shape[0] / overall_elapsed:.0f} papers/second overall")


if __name__ == '__main__':
    main()