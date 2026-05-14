import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from cs336_basics.transformer import AdamW, load_checkpoint, save_checkpoint, build_transformer_lm, cross_entropy_loss, get_batch
# ── arg parsing ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train-path", required=True)
    p.add_argument("--val-path", required=True)
    p.add_argument("--out-dir", default="checkpoints")
    # model
    p.add_argument("--vocab-size",      type=int, default=50257)
    p.add_argument("--context-length",  type=int, default=1024)
    p.add_argument("--d-model",         type=int, default=768)
    p.add_argument("--num-layers",      type=int, default=12)
    p.add_argument("--num-heads",       type=int, default=12)
    p.add_argument("--d-ff",            type=int, default=3072)
    p.add_argument("--rope-theta",      type=float, default=10000.0)
    # optimizer
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.999)
    p.add_argument("--eps",          type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    # training
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--max-steps",       type=int,   default=10000)
    p.add_argument("--log-interval",    type=int,   default=100)
    p.add_argument("--val-interval",    type=int,   default=500)
    p.add_argument("--val-steps",       type=int,   default=20)
    p.add_argument("--ckpt-interval",   type=int,   default=1000)
    p.add_argument("--resume",          type=str,   default=None)
    p.add_argument("--device",          type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb",           action="store_true")
    p.add_argument("--wandb-project",   type=str,   default="transformer-lm")
    return p.parse_args()

# ── data ───────────────────────────────────────────────────────────────────────

def load_memmap(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")

def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    inputs = np.stack([data[i : i + context_length]     for i in starts])
    targets = np.stack([data[i + 1 : i + context_length + 1] for i in starts])
    inputs = torch.tensor(inputs,  dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs, targets

# ── lr schedule (cosine with warmup) ──────────────────────────────────────────

def get_lr(step: int, max_steps: int, lr: float, warmup_steps: int = 100) -> float:
    if step < warmup_steps:
        return lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return lr * 0.5 * (1 + np.cos(np.pi * progress))


# ── validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(val_data, weights, args) -> float:
    losses = []
    for _ in range(args.val_steps):
        inputs, targets = get_batch(val_data, args.batch_size, args.context_length, args.device)
        logits = build_transformer_lm(
            args.vocab_size, args.context_length, args.d_model,
            args.num_layers, args.num_heads, args.d_ff,
            args.rope_theta, weights, inputs,
        )
        loss = CrossEntropyLoss(logits.view(-1, args.vocab_size), targets.view(-1))
        losses.append(loss.item())
    return float(np.mean(losses))

# ── decoder ─────────────────────────────────────────────────────────────────────
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt)
    tokens = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    eos_id = tokenizer.encode("<|endoftext|>")[0]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to context length
            context = tokens[:, -model.context_length :]

            logits = model(context)  # (1, T, vocab_size)
            logits = logits[:, -1, :]  # last token → (1, vocab_size)

            # temperature scaling
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)  # greedy
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)  # (1, vocab_size)

                # top-p (nucleus) sampling
                if top_p < 1.0:
                    probs = top_p_filter(probs, top_p)

                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            tokens = torch.cat([tokens, next_token], dim=-1)

            if next_token.item() == eos_id:
                break

    generated_ids = tokens[0, len(input_ids) :].tolist()
    return tokenizer.decode(generated_ids)


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    # probs: (1, vocab_size)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # remove tokens where cumulative prob exceeds top_p
    # shift right by 1 so we keep the token that pushes cumsum over top_p
    remove = (cumulative - sorted_probs) >= top_p
    sorted_probs[remove] = 0.0

    # scatter back to original ordering
    probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize
    return probs


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    train_data = load_memmap(args.train_path)
    val_data = load_memmap(args.val_path)

    # Use implemented build_transformer_lm to build the model
    model = build_transformer_lm(
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
        start_step, _ = load_checkpoint(args.resume, optimizer)
        print(f"resumed from step {start_step}")

    t0 = time.time()
    for step in range(start_step, args.max_steps):
        # lr schedule
        lr = get_lr(step, args.max_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(inputs)                                          # (B, T, vocab)
        loss   = cross_entropy_loss(logits.view(-1, args.vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ── logging ───────────────────────────────────────────────────────────
        if step % args.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")
            if args.wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
            t0 = time.time()

        # ── validation ────────────────────────────────────────────────────────
        if step % args.val_interval == 0:
            val_loss = evaluate(val_data, {n: p for n, p in model.named_parameters()}, args)
            print(f"  val loss {val_loss:.4f} | ppl {np.exp(val_loss):.2f}")
            if args.wandb:
                wandb.log({"val/loss": val_loss, "val/ppl": np.exp(val_loss)}, step=step)

        # ── checkpoint ────────────────────────────────────────────────────────
        if step % args.ckpt_interval == 0 and step > 0:
            save_checkpoint(
                f"{args.out_dir}/ckpt_{step:06d}.pt",
                step,
                model.state_dict(),
                optimizer.state_dict(),
                args,
            )

    # final checkpoint
    save_checkpoint(f"{args.out_dir}/ckpt_final.pt", args.max_steps, model.state_dict(), optimizer.state_dict(), args)

if __name__ == "__main__":
    main()
