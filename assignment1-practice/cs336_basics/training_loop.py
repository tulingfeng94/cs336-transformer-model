import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
<<<<<<< HEAD

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.transformer import (
    AdamW,
    CrossEntropyLoss,
    TransformerLM,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)

=======
from pathlib import Path
from cs336_basics.transformer import AdamW, load_checkpoint, save_checkpoint, build_transformer_lm, CrossEntropyLoss, get_batch
>>>>>>> 52171c4 (Assignment2 systems: DDP, FSDP, benchmarks)
# ── arg parsing ────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Transformer LM on pre-tokenized .bin memmaps. "
        "Run `python -m cs336_basics.bpe_tokenizer` first to build vocab + .bin files.",
    )
    p.add_argument("--train-path", required=True, help="uint16 memmap (.bin) from bpe_tokenizer")
    p.add_argument("--val-path", required=True, help="uint16 memmap (.bin) from bpe_tokenizer")
    p.add_argument(
        "--tokenizer-dir",
        default="data",
        help="directory with vocab.json (used to set vocab_size unless --vocab-size is set)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="override vocab size (default: len(vocab.json) in --tokenizer-dir)",
    )
    p.add_argument("--out-dir", default="checkpoints")
    # model
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # optimizer
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--val-interval", type=int, default=500)
    p.add_argument("--val-steps", type=int, default=20)
    p.add_argument("--ckpt-interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="transformer-lm")
    return p.parse_args()


# ── data ───────────────────────────────────────────────────────────────────────


def load_memmap(path: str | Path) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def vocab_size_from_tokenizer_dir(tokenizer_dir: Path) -> int:
    """Number of token IDs (embedding rows), not len(vocab.json) string keys.

    GPT-2-style vocab.json maps display string -> id; duplicate byte sequences
    collapse to one key, so len(json) can be far smaller than the true vocab size.
    """
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"missing {vocab_path}. Run tokenization first:\n"
            "  uv run python -m cs336_basics.bpe_tokenizer --train-txt ... --corpus-txt ..."
        )
    with vocab_path.open(encoding="utf-8") as f:
        vocab_gpt2 = json.load(f)
    if vocab_gpt2:
        from_ids = max(int(i) for i in vocab_gpt2.values()) + 1
    else:
        from_ids = 0
    if merges_path.is_file():
        merge_count = sum(
            1
            for line in merges_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and len(line.split()) == 2
        )
        # 1 special + 256 bytes + merges (see generate_vocab in bpe_tokenizer.py)
        from_merges = 1 + 256 + merge_count
        return max(from_ids, from_merges)
    return from_ids


def assert_token_ids_in_range(data: np.ndarray, vocab_size: int, name: str) -> None:
    max_id = int(np.max(data)) if len(data) else 0
    if max_id >= vocab_size:
        raise ValueError(
            f"{name}: max token id {max_id} >= vocab_size {vocab_size}. "
            f"Pass --vocab-size {max_id + 1} (or re-tokenize with matching BPE)."
        )


def load_datasets(args) -> tuple[np.ndarray, np.ndarray]:
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    for path in (train_path, val_path):
        if path.suffix != ".bin":
            raise ValueError(
                f"{path} is not a .bin memmap. Tokenize first:\n"
                "  uv run python -m cs336_basics.bpe_tokenizer \\\n"
                "    --train-txt data/TinyStoriesV2-GPT4-train.txt \\\n"
                "    --corpus-txt data/TinyStoriesV2-GPT4-train.txt \\\n"
                "    --corpus-txt data/TinyStoriesV2-GPT4-valid.txt \\\n"
                "    --out-dir data"
            )
    print(f"loading train memmap {train_path}")
    print(f"loading val memmap {val_path}")
    return load_memmap(train_path), load_memmap(val_path)


# ── lr schedule (cosine with warmup) ──────────────────────────────────────────


def get_lr(step: int, max_steps: int, lr: float, warmup_steps: int = 100) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return lr * 0.5 * (1 + np.cos(np.pi * progress))


# ── validation ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: TransformerLM, val_data, args) -> float:
    model.eval()
    losses = []
    for _ in range(args.val_steps):
        inputs, targets = get_batch(val_data, args.batch_size, args.context_length, args.device)
        logits = model(inputs)
        loss = CrossEntropyLoss(logits.view(-1, args.vocab_size), targets.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.vocab_size is None:
        args.vocab_size = vocab_size_from_tokenizer_dir(Path(args.tokenizer_dir))

    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    train_data, val_data = load_datasets(args)
    assert_token_ids_in_range(train_data, args.vocab_size, str(args.train_path))
    assert_token_ids_in_range(val_data, args.vocab_size, str(args.val_path))
    print(f"vocab_size={args.vocab_size} | train tokens={len(train_data):,} | val tokens={len(val_data):,}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"resumed from step {start_step}")

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.max_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(inputs)
        loss = CrossEntropyLoss(logits.view(-1, args.vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
            t0 = time.time()

        if step % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args)
            print(f"  val loss {val_loss:.4f} | ppl {np.exp(val_loss):.2f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss, "val/ppl": np.exp(val_loss)}, step=step)

        if step % args.ckpt_interval == 0 and step > 0:
            save_checkpoint(
                f"{args.out_dir}/ckpt_{step:06d}.pt",
                model,
                optimizer,
                step,
            )

    save_checkpoint(f"{args.out_dir}/ckpt_final.pt", model, optimizer, args.max_steps)


if __name__ == "__main__":
    main()
