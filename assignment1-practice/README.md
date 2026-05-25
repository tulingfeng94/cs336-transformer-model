# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## End-to-end workflow (tokenize → train → generate)

Training reads **pre-tokenized** `uint16` memmaps (`.bin`). BPE training and `.txt` → `.bin` conversion live in `cs336_basics/bpe_tokenizer.py`. LM training and checkpointing live in `cs336_basics/training_loop.py`. Sampling lives in `cs336_basics/generate_text.py`.

Run everything from this directory:

```sh
cd assignment1-practice
```

### 1. Tokenization (BPE + `.bin` memmaps)

**Full run** — train BPE on the train corpus, then tokenize train and valid:

```sh
uv run python -m cs336_basics.bpe_tokenizer \
  --train-txt data/TinyStoriesV2-GPT4-train.txt \
  --corpus-txt data/TinyStoriesV2-GPT4-train.txt \
  --corpus-txt data/TinyStoriesV2-GPT4-valid.txt \
  --out-dir data \
  --vocab-size 10000 \
  --pretoken-num-workers 16 \
  --tokenize-num-workers 16 \
  2>&1 | tee data/tokenize.log
```

This writes:

| Artifact | Purpose |
|----------|---------|
| `data/vocab.json`, `data/merges.txt` | BPE vocabulary (for tokenizer + `vocab_size`) |
| `data/TinyStoriesV2-GPT4-train.bin` | Train token memmap |
| `data/TinyStoriesV2-GPT4-valid.bin` | Validation token memmap |

**Re-tokenize only** (vocab/merges already exist):

```sh
uv run python -m cs336_basics.bpe_tokenizer \
  --train-txt data/TinyStoriesV2-GPT4-train.txt \
  --corpus-txt data/TinyStoriesV2-GPT4-train.txt \
  --corpus-txt data/TinyStoriesV2-GPT4-valid.txt \
  --out-dir data \
  --skip-train \
  --tokenize-num-workers 16
```

Tokenization is CPU-heavy; use `tmux` for long runs. Confirm `train.bin` is non-empty before training:

```sh
ls -lh data/*.bin
```

**`vocab_size` for the LM** must match BPE (e.g. `10000` for `--vocab-size 10000`). Do not use `len(vocab.json)` alone — duplicate byte strings collapse in the JSON file. `training_loop` infers size from `vocab.json` + `merges.txt` unless you pass `--vocab-size`.

### 2. Training

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_basics.training_loop \
  --train-path data/TinyStoriesV2-GPT4-train.bin \
  --val-path data/TinyStoriesV2-GPT4-valid.bin \
  --tokenizer-dir data \
  --max-steps 10000 \
  --batch-size 32 \
  --context-length 1024 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --device cuda
```

Checkpoints are saved under `checkpoints/` every `--ckpt-interval` steps (default `1000`), plus `ckpt_final.pt` at the end.

**Resume / reload training** from a checkpoint (same model flags as the original run):

```sh
CUDA_VISIBLE_DEVICES=0 uv run python -m cs336_basics.training_loop \
  --train-path data/TinyStoriesV2-GPT4-train.bin \
  --val-path data/TinyStoriesV2-GPT4-valid.bin \
  --tokenizer-dir data \
  --resume checkpoints/ckpt_005000.pt \
  --max-steps 10000 \
  --batch-size 32 \
  --context-length 1024 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --device cuda
```

Training continues from the saved step through `--max-steps`. Use an earlier numbered checkpoint (`ckpt_001000.pt`, …) or `ckpt_final.pt` as needed.

### 3. Text generation

`cs336_basics/generate_text.py` loads a checkpoint and samples with **`--temperature`** and **`--top-p`** from the command line.

**Default sampling** (temperature `0.8`, top-p `0.95`):

```sh
uv run python -m cs336_basics.generate_text \
  --checkpoint checkpoints/ckpt_final.pt \
  --tokenizer-dir data \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top-p 0.95 \
  --min-new-tokens 256 \
  --max-new-tokens 512 \
  --output generation_dump.txt \
  --device cuda
```

**Decoder flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | `0.8` | Sampling temperature; `0` = greedy argmax |
| `--top-p` | `0.95` | Nucleus (top-p) cutoff in `(0, 1]`; `1.0` = no top-p filter |

**Greedy decoding** (deterministic):

```sh
uv run python -m cs336_basics.generate_text \
  --checkpoint checkpoints/ckpt_final.pt \
  --prompt "Once upon a time" \
  --temperature 0 \
  --top-p 1.0
```

**Length and stopping**

| Flag | Default | Description |
|------|---------|-------------|
| `--min-new-tokens` | `256` | Block `<|endoftext|>` until this many tokens are generated |
| `--max-new-tokens` | `512` | Hard cap on new tokens |
| `--ignore-eos` | off | Never stop on `<|endoftext|>`; only `--max-new-tokens` limits length |

TinyStories-trained models often emit EOS early. For longer samples, use an open-ended prompt, raise `--max-new-tokens`, and optionally `--ignore-eos`:

```sh
uv run python -m cs336_basics.generate_text \
  --checkpoint checkpoints/ckpt_final.pt \
  --prompt "Once upon a time, there was a little girl who" \
  --temperature 0.8 \
  --top-p 0.95 \
  --min-new-tokens 256 \
  --max-new-tokens 1024 \
  --ignore-eos \
  --output generation_dump.txt \
  --device cuda
```

**Other flags**

- **`--checkpoint`**: any saved checkpoint (`ckpt_003000.pt`, `ckpt_final.pt`, …).
- **Model flags** (`--d-model`, `--num-layers`, `--num-heads`, `--d-ff`, `--context-length`) must match how that checkpoint was trained (defaults above match the training example in §2).

