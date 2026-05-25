"""
Generate text from a trained Transformer LM checkpoint.

Decoder sampling: ``--temperature`` (0 = greedy) and ``--top-p`` (nucleus sampling).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.training_loop import vocab_size_from_tokenizer_dir


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering on the last dimension."""
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = (cumulative - sorted_probs) >= top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    blocked_ids: set[int] | None = None,
) -> int:
    """
    Sample one token from logits (..., vocab).

    - temperature=0: greedy argmax
    - temperature>0: softmax(logits / T), optional top-p, then multinomial
    """
    logits = logits.clone()
    if blocked_ids:
        for tid in blocked_ids:
            logits[..., tid] = -float("inf")

    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())

    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_token_ids(
    model: TransformerLM,
    prompt_ids: list[int],
    *,
    eos_id: int,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    ignore_eos: bool = False,
) -> list[int]:
    """Sample until ``max_new_tokens``. EOS ends early unless masked or ``ignore_eos``."""
    model.eval()
    tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        context = tokens[:, -model.context_length :]
        logits = model(context)[:, -1, :]

        blocked: set[int] = set()
        if ignore_eos or len(generated) < min_new_tokens:
            blocked.add(eos_id)

        next_id = sample_next_token(
            logits,
            temperature=temperature,
            top_p=top_p,
            blocked_ids=blocked or None,
        )

        generated.append(next_id)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

        if next_id == eos_id:
            break

    return generated


def validate_sampling(temperature: float, top_p: float) -> None:
    if temperature < 0.0:
        raise ValueError(f"--temperature must be >= 0, got {temperature}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"--top-p must be in (0, 1], got {top_p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate text from a Transformer LM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ckpt_final.pt"),
        help="path to .pt checkpoint from training_loop",
    )
    p.add_argument("--tokenizer-dir", type=Path, default=Path("data"))
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--prompt", type=str, default="Once upon a time")

    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature; 0 = greedy argmax",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus sampling cutoff in (0, 1]; 1.0 disables top-p filtering",
    )

    p.add_argument(
        "--min-new-tokens",
        type=int,
        default=256,
        help="block <|endoftext|> until this many tokens are generated",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="hard cap on generated tokens",
    )
    p.add_argument(
        "--ignore-eos",
        action="store_true",
        help="never stop on <|endoftext|>; only --max-new-tokens limits length",
    )
    p.add_argument("--output", type=Path, default=None, help="optional path to write full text + metadata")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    return p.parse_args()


def load_model_weights(checkpoint_path: Path, model: TransformerLM) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint.get("iteration", -1))


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < args.min_new_tokens:
        raise ValueError("--max-new-tokens must be >= --min-new-tokens")

    validate_sampling(args.temperature, args.top_p)

    vocab_size = args.vocab_size or vocab_size_from_tokenizer_dir(args.tokenizer_dir)
    tokenizer = BPETokenizer.from_files(
        args.tokenizer_dir / "vocab.json",
        args.tokenizer_dir / "merges.txt",
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    step = load_model_weights(args.checkpoint, model)

    prompt_ids = tokenizer.encode(args.prompt)
    gen_ids = generate_token_ids(
        model,
        prompt_ids,
        eos_id=tokenizer.eot_id,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        ignore_eos=args.ignore_eos,
    )

    generated_text = tokenizer.decode(gen_ids)
    full_text = args.prompt + generated_text
    stopped_on_eos = bool(gen_ids) and gen_ids[-1] == tokenizer.eot_id

    header = "\n".join(
        [
            "=== Generation ===",
            f"checkpoint: {args.checkpoint} (step {step})",
            f"prompt: {args.prompt!r}",
            f"temperature: {args.temperature}",
            f"top_p: {args.top_p}",
            f"ignore_eos: {args.ignore_eos}",
            f"generated_tokens: {len(gen_ids)} (min={args.min_new_tokens}, max={args.max_new_tokens})",
            f"stopped_on_<|endoftext|>: {stopped_on_eos}",
            "",
            "=== Text ===",
        ]
    )
    body = full_text + "\n"
    report = header + "\n" + body

    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
