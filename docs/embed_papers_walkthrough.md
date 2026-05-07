# SPECTER2 Embedding Pipeline — Complete Walkthrough

A reference document for `scripts/embed_papers.py`. Read top-to-bottom for understanding. Use as study material before whiteboarding.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Architecture Diagrams](#2-architecture-diagrams)
3. [Inside SPECTER2 — Deep Dive](#3-inside-specter2--deep-dive)
4. [The [CLS] Token Explained](#4-the-cls-token-explained)
5. [CPU vs GPU Memory](#5-cpu-vs-gpu-memory)
6. [The Script — Section by Section](#6-the-script--section-by-section)
7. [Why Specific Engineering Choices](#7-why-specific-engineering-choices)
8. [What to Whiteboard](#8-what-to-whiteboard)
9. [What Makes This Script Professional](#9-what-makes-this-script-professional)

---

## 1. The Big Picture

**What we're trying to do:** turn 902,645 abstracts (text) into 902,645 vectors (numbers a computer can compare).

**Why:** because text is hard for a computer to compare directly. Numbers are easy. If two papers about computer vision both produce vectors that point in similar directions in 768-dimensional space, the computer can find that match instantly. It can't do that with raw text.

**The model doing the conversion:** SPECTER2. A pre-trained transformer (think: a smaller cousin of GPT) that someone else trained on millions of scientific papers. It already knows what scientific writing "means." We're just using it to encode our papers.

**The output:** a giant table of numbers, 902,645 rows by 768 columns, saved as a `.npy` file. About 2.7 GB.

That's the whole project in 4 sentences. Everything else in the script is engineering to make that happen reliably.

---

## 2. Architecture Diagrams

### Diagram 1: Data flow (from parquet to embeddings)

```
        ┌──────────────────────────────────────────────┐
        │  arxiv_cs_clean.parquet  (902,645 papers)    │
        │  columns: id, title, abstract, year, ...     │
        └────────────────────┬─────────────────────────┘
                             │
                             ▼
              build "title [SEP] abstract"
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │  list of 902,645 strings (Python list)       │
        └────────────────────┬─────────────────────────┘
                             │
                  split into chunks of 50k
                             │
        ┌──────────────────────────────────────────────┐
        │  one chunk = 50,000 strings                  │
        └────────────────────┬─────────────────────────┘
                             │
                  inside chunk: model.encode()
                  splits into batches of 32
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │  one batch = 32 strings → SPECTER2 → GPU     │
        │            ↓                                 │
        │  output: 32 vectors of 768 floats            │
        └────────────────────┬─────────────────────────┘
                             │
                  collect all batches
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │  numpy array (50000, 768) for the chunk      │
        └────────────────────┬─────────────────────────┘
                             │
                save chunk into big array slot
              [chunk_start : chunk_end] = chunk_embeddings
                             │
                             ▼
        ┌──────────────────────────────────────────────┐
        │  big array  (902645, 768)  float32  ~2.7 GB  │
        └────────────────────┬─────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        embeddings.npy            papers_meta.parquet
        (the vectors)             (which row = which paper)
                ▼                         ▼
        used by V3 Day 2+ for FAISS index, recommender, RAG
```

**Two key concepts:**
- **Chunk = checkpoint unit (50,000 papers).** You crash, you lose at most one chunk's worth of work.
- **Batch = GPU parallelism unit (32 papers).** GPU processes 32 at the same time using parallel matrix math.

### Diagram 2: What happens inside SPECTER2 for ONE paper

```
Input: "Sparsity-certifying Graph Decompositions [SEP] We describe a new..."
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │  STEP 1: Tokenization                  │
        │  Split text into "tokens" (word-pieces)│
        │  Convert each token to a number (ID)   │
        └────────────────────┬───────────────────┘
                             │
                             ▼
              [101, 8329, 1011, 17708, 102, 2057, 6235, ...]
              (a list of ~256 integers)
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 2: Embedding lookup              │
        │  Each token ID → a 768-number vector   │
        │  (the model learned these in training) │
        └────────────────────┬───────────────────┘
                             │
                             ▼
              matrix of shape (256, 768)
              ↑ one row per token, 768 numbers per row
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 3: 12 Transformer layers         │
        │  Each layer:                           │
        │    a) Attention: tokens "look at" each │
        │       other and update their meanings  │
        │    b) Feedforward: each token's vector │
        │       passes through a small neural net│
        └────────────────────┬───────────────────┘
                             │
                             ▼
              still shape (256, 768), but values changed
              every token's vector now reflects
              the meaning of the whole paper
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 4: Take the [CLS] token's vector │
        │  (the very first row of the matrix)    │
        └────────────────────┬───────────────────┘
                             │
                             ▼
                a single vector of 768 numbers
                = "this paper, condensed"
                             │
                             ▼
                Output: shape (768,) — one paper, one vector
```

---

## 3. Inside SPECTER2 — Deep Dive

A transformer layer has two halves: **attention** and **feedforward**. Both modify the (256, 768) matrix in place — same shape in, same shape out, different numbers.

### The attention half — "tokens look at each other"

Imagine the abstract is a meeting with 256 people sitting around a table. Each person has 768 facts about themselves (their token vector). Attention is a structured conversation:

For every person at the table:
1. They write down a **query**: "what kind of information am I looking for?"
2. Every other person writes down a **key**: "this is what I know about."
3. The person compares their query against everyone else's keys. Whoever's key matches best, the person pays the most "attention" to.
4. They then collect a weighted average of everyone's **value** (their actual content), with more weight on the people whose keys matched their query.
5. They update their own 768 numbers based on what they learned.

**Concrete example.** Take the token "image" in an abstract about computer vision. Its query might be: "what am I being applied to?" Other tokens in the same abstract — "segmentation", "neural", "network" — have keys that match this query well. So "image" updates its vector to become more like "image-in-the-context-of-neural-segmentation" rather than just generic "image."

After this step, every token's 768 numbers have shifted to reflect not just *the token itself* but *the token in the context of every other token in the abstract*.

That's why transformers are powerful: the same word means different things in different contexts, and attention captures that.

**The math underneath: it's all matrix multiplications.**

```
Q = X @ W_q   # queries:  (256, 768) × (768, 768) → (256, 768)
K = X @ W_k   # keys:     same shape
V = X @ W_v   # values:   same shape

scores = Q @ K.T / sqrt(768)   # (256, 256) — every token vs every other
weights = softmax(scores)      # turn into probabilities
output = weights @ V            # (256, 768) — weighted average of values
```

`W_q`, `W_k`, `W_v` are learned weight matrices. Pre-training is the process of finding the values that make this work for scientific text.

The GPU is fast at this because every step is matrix multiplication, which GPUs were literally built for. Doing 32 papers at once (batch_size=32) means doing matrix multiplications on tensors of shape (32, 256, 768) — same operations, just bigger matrices, almost no extra wall-clock time per paper.

### The feedforward half — "each token thinks for itself"

After attention, each token has a context-aware 768-vector. The feedforward step then says: "OK, now process that vector."

For every token independently:
1. Take its 768 numbers.
2. Pass through a small fully-connected neural network: 768 → 3072 → 768.
3. Replace the original 768 numbers with the new 768 numbers.

**Why bigger in the middle (3072):** the larger hidden layer lets the network represent more complex transformations. Standard architectural choice from BERT.

This is where the model can apply learned patterns to individual tokens: "if this token in this context has these properties, transform it like so."

**Crucially: attention has the tokens *talking to each other*; feedforward has each token *thinking on its own*.** Together they make a "transformer layer."

### Stack 12 of these layers

After 12 rounds of attention + feedforward, every token's 768 numbers have been refined 12 times. By the last layer, each token's vector encodes not just the token, not just its immediate neighbors, but its role in the entire paper's meaning.

**Why 12?** It's a hyperparameter. BERT-base (which SPECTER2 is built on) uses 12. Bigger versions (BERT-large, 24 layers) are more powerful but slower. 12 is the sweet spot for the model size SPECTER2 was trained at.

---

## 4. The [CLS] Token Explained

When you tokenize text for BERT-family models, the tokenizer prepends a special token: `[CLS]` (short for "classification"). Its token ID is 101 in BERT's vocabulary.

```
Original:    "Sparsity-certifying Graph Decompositions [SEP] We describe..."
Tokenized:   [CLS] Sparsity ##-cert ##ifying Graph Decompos ##itions [SEP] We describe ...
Token IDs:   [101, 8329, 1011, ..., 102, 2057, 6235, ...]
```

The `[CLS]` token has no semantic meaning on its own. It's a *placeholder* the model learns to use as a "summary slot."

**Here's the trick:** during training, the model learned that whatever ends up in the [CLS] token's final vector should represent the entire input. Because attention lets every token gather information from every other token, by the final layer, the [CLS] token has "looked at" every word in the paper and collected information from all of them.

So when we want one vector per paper, we don't average all 256 token vectors (which would dilute information). We just pull out the [CLS] token's vector. That's our paper embedding.

**Concrete picture:**

```
After 12 transformer layers, the matrix looks like:

     position  token        768-dim vector after processing
       0       [CLS]     →  [0.12, -0.45, 0.88, ..., 0.03]   ← we take this
       1       Sparsity  →  [0.34, 0.11, -0.22, ..., 0.91]
       2       ##-cert   →  [0.55, -0.08, 0.16, ..., 0.43]
       ...
       255     [PAD]     →  [0.00, 0.00, 0.00, ..., 0.00]

The [CLS] vector at position 0 is what we save.
```

The line in the script that does this is hidden inside `model.encode()`. SentenceTransformer wraps it. If you wrote it manually with raw HuggingFace it would look like:

```python
output = model(input_ids, attention_mask=...)
hidden_states = output.last_hidden_state    # (batch_size, 256, 768)
cls_vectors = hidden_states[:, 0, :]         # take row 0 from each item → (batch_size, 768)
```

That `[:, 0, :]` slice is the "take the [CLS] token" step. Position 0 in the token sequence.

---

## 5. CPU vs GPU Memory

Your computer has two separate memory pools:

- **CPU RAM** (the regular memory, ~16-32 GB on a laptop). Where Python lives.
- **GPU VRAM** (separate memory chip on the graphics card, 8 GB on your 4060). Where neural network computations happen.

Data has to be explicitly moved between them. The GPU can't see CPU RAM, and vice versa.

For each batch of 32 papers, the cycle is:

```
CPU RAM (Python list of strings)
       │
       │  1. Tokenize on CPU (turn strings into integer IDs)
       │
       │  2. Move integer IDs from CPU to GPU
       │     (this is the .to('cuda') call)
       ▼
GPU VRAM (integer IDs)
       │
       │  3. Run all the transformer math on GPU
       │     (12 layers of attention + feedforward)
       │
       ▼
GPU VRAM (32 vectors of 768 floats)
       │
       │  4. Move vectors from GPU back to CPU
       │     (this is .cpu().numpy())
       ▼
CPU RAM (numpy array, 32 × 768)
       │
       │  5. Stuff into the big embeddings array
       │     embeddings[chunk_start:chunk_end] = chunk_embeddings
       ▼
CPU RAM (the big 902645 × 768 array)
```

**Why this matters:** the **transfer between CPU and GPU is slow** compared to the math itself. This is why batching matters. If you sent 1 paper at a time, you'd pay the transfer cost 902,645 times. By batching 32 at once, you pay it 28,000 times (32× less). At batch_size=64 (if your VRAM allowed), it'd be 14,000 transfers.

**Bigger batches = fewer round trips = faster overall.** The limit is GPU memory: 32 papers × 256 tokens × 768 dims × intermediate computations = roughly 2-3 GB of VRAM. Your 8 GB card has headroom; we could probably go to batch_size=64 if we wanted, but 32 is safe.

This is also why **the speed test was slow per-paper but the real run is fast.** The speed test had transfer overhead averaged over only 1000 papers. The real run amortizes startup costs over 902,645 papers.

---

## 6. The Script — Section by Section

### Section 1: Imports and configuration

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
```

Plain English:
- **`Path`** — modern way to write file paths that works on Windows, Mac, Linux without breaking
- **`numpy`** — fast number arrays. The 902k × 768 matrix lives in numpy
- **`pandas`** — for reading the parquet file into a table
- **`torch`** — PyTorch. Talks to your GPU
- **`SentenceTransformer`** — the wrapper around SPECTER2 that gives us the simple `model.encode(text)` interface
- **`tqdm`** — the live progress bar you see in the terminal

```python
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'arxiv_cs_clean.parquet'
EMBEDDINGS_PATH = MODELS_DIR / 'v3_specter2_embeddings.npy'
```

Plain English: "find where this script lives on disk, then build paths relative to that." The reason: if you run the script from `notebooks/` or from the project root, it still finds the data file. No "working directory" confusion.

```python
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 256
CHECKPOINT_EVERY = 50_000
```

These are the **dials** of the script:
- **batch_size = 32** — how many papers we send to the GPU at once. Bigger = faster but uses more VRAM. 32 fits in your 8 GB comfortably. If you saw "out of memory" errors, you'd lower this to 16 or 8.
- **max_seq_length = 256** — how many word-pieces from each abstract the model looks at. SPECTER2 was trained at 256, so we match that. Longer = more memory, no gain in quality.
- **checkpoint_every = 50_000** — every 50,000 papers, dump current progress to disk. So if you crash, you lose at most 50k of work, not all 900k.

### Section 2: Helpers — small reusable functions

```python
def get_gpu_temp():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', ...])
    return int(result.stdout.strip().split('\n')[0])
```

Plain English: "ask the GPU driver what temperature the GPU is right now, return a number like 72." Lets us notice if the laptop is overheating.

```python
def format_duration(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    if seconds < 3600: return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"
```

Plain English: "convert raw seconds into something human-readable." 4500 seconds becomes "1.3h" instead of "4500s". Pure cosmetics.

```python
def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)
```

Plain English: "print a message with a timestamp." That's why every line in your terminal looks like `[16:41:29] Model loaded`. The `flush=True` part means "show it immediately, don't wait."

### Section 3: The speed test

```python
def run_speed_test(model, abstracts, batch_size):
    t0 = time.time()
    _ = model.encode(abstracts, batch_size=batch_size, ...)
    elapsed = time.time() - t0
    speed = len(abstracts) / elapsed
    return speed
```

Plain English: "embed 1000 papers, time it, return how fast we went."

**Why:** before committing to a 5-hour run, we want a number to plan around. If the speed test had said "0.5 papers/second," that would be a sign something is wrong — maybe the GPU isn't being used. We'd want to investigate before letting it run for days.

The `_ =` part means "throw away the result." We don't keep these embeddings — they get re-embedded in the real run. We just wanted the timing.

### Section 4: The main embedding loop (the heart of the script)

```python
def embed_corpus(model, abstracts, batch_size, start_index=0, existing_embeddings=None):
    n_total = len(abstracts)
    embeddings = np.zeros((n_total, EMBEDDING_DIM), dtype=np.float32)
```

Plain English: "make a big empty table of zeros, 902,645 rows × 768 columns. We'll fill it in as we go."

The `dtype=np.float32` matters: 32-bit floats use 4 bytes each. 902,645 × 768 × 4 bytes = 2.77 GB. If we used float64 (the default), it'd be 5.5 GB. Half the memory for no real loss in quality. Standard practice for ML embeddings.

```python
    if existing_embeddings is not None and start_index > 0:
        embeddings[:start_index] = existing_embeddings
```

Plain English: "if we're resuming after a crash, copy the saved partial results into our table first."

This is the **resume logic**. Say the script crashed at paper 200,000. We saved a checkpoint there. Now when we restart, this line copies those 200,000 rows into the new empty table, and we continue from row 200,001 instead of restarting.

```python
    pbar = tqdm(total=n_total, initial=start_index, ...)
```

Plain English: "set up the progress bar." `total` is the goal (902,645). `initial=start_index` means "if we're resuming from 200k, start the bar at 200k, not 0."

#### The actual loop

```python
    chunk_size = CHECKPOINT_EVERY  # 50,000

    for chunk_start in range(start_index, n_total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_total)
        chunk = abstracts[chunk_start:chunk_end]
```

Plain English: "process papers 50,000 at a time. Get a slice."

So the loop runs ~18 times total (902,645 ÷ 50,000 ≈ 18). Each iteration handles 50k papers.

**Wait — but `BATCH_SIZE = 32`, not 50,000. What's the difference?**

- **Chunk = 50,000.** This is the *checkpointing* unit. Every 50k papers, we save to disk.
- **Batch = 32.** This is the *GPU* unit. We send 32 papers to the GPU at once for parallel processing.

So one chunk of 50,000 contains ~1,562 batches of 32. The next line handles the batching:

```python
        chunk_embeddings = model.encode(
            chunk,
            batch_size=batch_size,    # 32 — internal GPU batching
            show_progress_bar=False,
            convert_to_numpy=True,
        )
```

Plain English: "give SPECTER2 these 50,000 abstracts. It'll internally split them into batches of 32, send each batch to the GPU, get back 32 vectors at a time, and stitch them together. Return all 50,000 vectors as a numpy array."

This single line is where 99% of the GPU work happens. Inside, SPECTER2:
1. Tokenizes each abstract (turns words into numbers the model understands)
2. Pads short abstracts and truncates long ones to 256 tokens
3. Sends the batch to GPU memory
4. Runs 12 layers of attention + feedforward computations
5. Pulls out the [CLS] token's final state (a 768-number summary)
6. Sends the result back to CPU memory

Repeated 1,562 times per chunk. ~28,000 times for the full corpus.

```python
        embeddings[chunk_start:chunk_end] = chunk_embeddings
        pbar.update(len(chunk))
```

Plain English: "stuff the results into our big table at the right rows. Tell the progress bar we did 50k more."

#### The temperature check

```python
        if chunk_end - last_gpu_check >= GPU_CHECK_EVERY:
            temp = get_gpu_temp()
            if temp is not None:
                pbar.set_postfix({'GPU_C': temp})
                if temp >= GPU_TEMP_WARN:
                    log(f"⚠️  GPU at {temp}°C — thermal throttling likely.")
            last_gpu_check = chunk_end
```

Plain English: "every 10,000 papers, ask the GPU its temperature. Show it next to the progress bar. If it's above 80°C, print a warning so the user knows to act."

**Why this matters:** laptop GPUs throttle when hot. If your 4060 hits 85°C, the GPU clock automatically drops, slowing the embedding run by 30-50%. Knowing you've hit thermal throttling lets you decide whether to pause.

#### The checkpoint save

```python
        if chunk_end - last_checkpoint >= CHECKPOINT_EVERY or chunk_end == n_total:
            log(f"Checkpoint: saving {chunk_end:,} embeddings...")
            np.save(CHECKPOINT_PATH, embeddings[:chunk_end])
            PROGRESS_PATH.write_text(str(chunk_end))
            last_checkpoint = chunk_end
```

Plain English: "every 50,000 papers, dump the partial table to disk along with a progress marker."

Two files get written:
- `v3_specter2_embeddings.checkpoint.npy` — the partial embeddings table
- `v3_embedding_progress.txt` — just a number, e.g. "200000", saying how far we got

**Why two files: redundancy.** If the .npy save partially succeeds and corrupts, the progress file might still say "200000" — but when we try to resume, we'll detect the shape mismatch and start over. If the progress file fails to write, we won't pick up the partial .npy. Belt and suspenders.

### Section 5: The main() function — the orchestrator

This is what runs first when you type `python scripts/embed_papers.py`.

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--restart', action='store_true')
    args = parser.parse_args()
```

Plain English: "let the user pass options on the command line." If you run `python scripts/embed_papers.py --batch-size 16`, the script uses 16 instead of 32. If you run with `--restart`, it ignores any saved checkpoint and starts fresh.

This is what makes a script *professional* instead of amateur — configurable from outside without editing the file.

#### The 8 steps in main()

**1. Verify environment**
```python
if not torch.cuda.is_available(): ...
```
"Check that we have a GPU. Warn if not."

**2. Load corpus**
```python
df = pd.read_parquet(DATA_PATH)
df['_input'] = df['title'] + ' [SEP] ' + df['abstract']
abstracts = df['_input'].tolist()
```
"Read the parquet, build the input strings, get them into a list."

The `[SEP]` thing is important. SPECTER2 was trained on `title [SEP] abstract` as input — the [SEP] tells the model "these are two related but distinct pieces of text." If you just concatenate them with a space, the model gets confused. This is a small detail with real quality impact.

**3. Resume detection**
```python
if not args.restart and CHECKPOINT_PATH.exists():
    saved_index = int(PROGRESS_PATH.read_text().strip())
    existing_embeddings = np.load(CHECKPOINT_PATH)
    start_index = saved_index
    log(f"Found checkpoint at {start_index:,} papers — resuming.")
```
"Check if there's a checkpoint from a previous run. If so, set up to resume from there."

**4. Load model**
```python
model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = args.max_seq_length
```
"Download SPECTER2 (or use cached copy), put it on the GPU, set the input length."

**5. Speed test**
```python
sample = abstracts[:1000]
speed = run_speed_test(model, sample, args.batch_size)
est_total_min = (len(abstracts) / speed) / 60
log(f"Estimated full run time: {est_total_min:.0f} minutes")
time.sleep(5)
```
"Run the warmup test, print the estimate, give the user 5 seconds to abort if they don't like the number."

**6. Full embedding run**
```python
all_embeddings = embed_corpus(model, abstracts, ...)
```
"The actual work. The function we already explained."

**7. Save final outputs**
```python
np.save(EMBEDDINGS_PATH, all_embeddings)
df[META_COLUMNS].to_parquet(META_PATH, index=False)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()  # delete checkpoint
```
"Save the final embeddings and metadata. Delete the checkpoint files since we don't need them anymore."

The metadata file is important: it's the "key" that tells you which row of the embedding matrix corresponds to which paper. Without it, you have 902k vectors but no way to know which is which.

**8. Summary**
```python
log(f"✓ Done in {format_duration(overall_elapsed)}")
log(f"  Papers embedded: {all_embeddings.shape[0]:,}")
log(f"  Avg speed: {all_embeddings.shape[0] / overall_elapsed:.0f} papers/second")
```
"Print final stats."

---

## 7. Why Specific Engineering Choices

### Why `convert_to_numpy=True`

```python
chunk_embeddings = model.encode(
    chunk,
    batch_size=batch_size,
    convert_to_numpy=True,    # ← this
)
```

By default, `model.encode` returns PyTorch tensors that *might still be on the GPU*. We want numpy arrays on the CPU, because:

1. The big `embeddings` array we're filling is a numpy array on CPU.
2. We need to write to disk, which only works with CPU data.
3. PyTorch tensors hold a reference to the GPU memory they came from — keeping them around means VRAM doesn't get freed for the next batch.

Setting `convert_to_numpy=True` does the GPU→CPU transfer immediately and gives us a clean numpy array. VRAM gets recycled for the next batch.

### Why `model.max_seq_length = 256`

```python
model.max_seq_length = args.max_seq_length    # 256
```

If an abstract is longer than 256 tokens, it gets **truncated** — anything past token 256 is silently dropped. This is fine for arxiv abstracts (most are 100-200 tokens), but it means a 500-word abstract loses some content.

If we set max_seq_length = 512, we'd capture more text, but:
- Memory per batch would roughly double (attention is quadratic in sequence length: 256² → 512² = 4× more work per attention computation)
- Speed would drop by 2-3x
- SPECTER2 was trained at 256, so longer sequences are extrapolation territory anyway — quality might not improve

256 is the right setting for SPECTER2 specifically. Different model, different limit.

### What `np.float32` saves us

```python
embeddings = np.zeros((n_total, EMBEDDING_DIM), dtype=np.float32)
```

NumPy's default float type is `float64` (8 bytes per number). We force `float32` (4 bytes). For 902,645 × 768 = 693 million numbers:

- float64: 693M × 8 = 5.5 GB on disk
- float32: 693M × 4 = 2.77 GB on disk

The precision difference doesn't matter for embeddings — we're going to compute cosine similarities and rank them. The 7-digit precision of float32 is more than enough. float64's 15-digit precision is wasted here.

This is a standard ML practice: **for embeddings, always float32 (or even float16 if memory is tight).**

---

## 8. What to Whiteboard

Structure it in three sections.

### Section A — The data pipeline (the easy part)

Draw Diagram 1. Explain: 902k abstracts go in, 902k vectors come out, plus a metadata file so we know which row is which paper. Chunks of 50k for checkpointing, batches of 32 for GPU parallelism.

### Section B — Inside one forward pass (the meat)

Draw Diagram 2. Walk through the 4 steps:

1. **Tokenization**: text becomes integer IDs (~256 of them per paper, with [CLS] in position 0)
2. **Embedding lookup**: each ID → 768-number vector (matrix shape now (256, 768))
3. **12 transformer layers**: each layer does attention (tokens look at each other and update their meanings) then feedforward (each token's vector goes through a small NN)
4. **Take [CLS]**: position 0 holds the summary of the whole paper. We extract that one row.

If asked **"why does the [CLS] token end up as a summary?"** — answer: "because during training, the model learned to use that position as a summary slot. Attention lets every token gather information from every other, and by the last layer, [CLS] has gathered information from everything."

### Section C — The engineering around it (the polish)

The script's safety nets:
- Speed test before commitment — know what you're signing up for
- Chunked checkpointing every 50k — crash recovery
- Resume detection on startup — pick up where you left off
- GPU temp monitoring — notice throttling
- CLI arguments — configurable without editing
- float32 instead of float64 — half the disk space, no quality loss
- Path resolution from `__file__` — works from any working directory

If asked **"why did you build all these safety nets for a one-time job?"** — answer: "because in real ML, no run is one-time. Models change, data changes, you re-run. And during the first run, things crash. Building the script this way makes it sustainable infrastructure, not throwaway code."

---

## 9. What Makes This Script Professional

If your alternance interviewer pulls up this code and asks "walk me through this," here's what they'd notice:

1. **Configuration at the top, code below.** Easy to find what's tunable.
2. **Functions for distinct concerns.** `get_gpu_temp()`, `run_speed_test()`, `embed_corpus()`, `main()` — each does one thing.
3. **Argparse for CLI options.** Not hardcoded values that need editing.
4. **Checkpointing + resume.** Anyone who's done long-running ML knows things crash. Building this in shows you've worked on real systems.
5. **GPU temp monitoring.** Shows you understand the *hardware*, not just the model.
6. **Path handling that works from anywhere.** `Path(__file__).resolve().parent` is the mark of someone who's been bitten by directory bugs.
7. **Type hints (`-> int | None`).** Modern Python style, helps IDEs and readers.
8. **Cleanup of intermediate files.** Final outputs only. No leftover checkpoint clutter.

Those eight things together are what separate "I wrote a script that worked" from "I wrote a script someone could maintain." When you describe this project in an interview, you can point at the script and explain *why* you built it this way. That's the difference between getting an interview and getting an offer.

---

## Quick Reference Card

**Numbers to remember:**
- 902,645 papers in corpus
- 768-dim vectors per paper
- 256 tokens max per input
- 32 papers per GPU batch
- 50,000 papers per checkpoint
- 12 transformer layers in SPECTER2
- 110M parameters in the model
- ~2.7 GB for final embeddings file (float32)

**Key formulas:**
- Attention: `softmax(Q @ K.T / sqrt(d)) @ V`
- Feedforward: `Linear(768→3072) → GELU → Linear(3072→768)`
- Paper vector: take row 0 (the [CLS] position) from the final layer output

**Pipeline:**
text → tokenize → embed → 12 layers → take [CLS] → 768-dim vector

**The output enables:**
- FAISS index for fast similarity search (V3 Day 2)
- Semantic recommender (V3 Day 2)
- RAG question-answering (V3 Days 3-4)
- Fine-tuning starting point (V3 Week 2)