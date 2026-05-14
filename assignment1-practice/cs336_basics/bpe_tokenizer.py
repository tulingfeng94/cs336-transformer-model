"""
Byte-level BPE training aligned with GPT-2 / course reference tests.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import regex as re

from tests.common import gpt2_bytes_to_unicode

# GPT-2 pretokenization pattern (handout / tiktoken-style)
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _unicode_to_byte() -> dict[str, int]:
    return {v: k for k, v in gpt2_bytes_to_unicode().items()}


def gpt2_symbols_to_raw_bytes(s: str) -> bytes:
    """Map a GPT-2 display string (each char one raw byte) back to raw bytes."""
    u2b = _unicode_to_byte()
    return bytes(u2b[c] for c in s)


def pretoken_to_symbol_tuple(pt: str) -> tuple[str, ...]:
    b2u = gpt2_bytes_to_unicode()
    return tuple(b2u[b] for b in pt.encode("utf-8"))


def merge_token_bytes(
    token_bytes: tuple[str, ...], best_pair: tuple[str, str]
) -> tuple[str, ...]:
    new_seq: tuple[str, ...] = ()
    i = 0
    while i < len(token_bytes):
        if (
            i + 1 < len(token_bytes)
            and token_bytes[i] == best_pair[0]
            and token_bytes[i + 1] == best_pair[1]
        ):
            new_seq += (best_pair[0] + best_pair[1],)
            i += 2
        else:
            new_seq += (token_bytes[i],)
            i += 1
    return new_seq


def _count_pairs(corpus: dict[tuple[str, ...], int]) -> defaultdict[tuple[str, str], int]:
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for token_bytes, count in corpus.items():
        for i in range(len(token_bytes) - 1):
            pair_counts[(token_bytes[i], token_bytes[i + 1])] += count
    return pair_counts


def update_pair_counts(
    pair_counts: defaultdict[tuple[str, str], int],
    best_pair: tuple[str, str],
    token_bytes: tuple[str, ...],
    count: int,
) -> None:
    merged = best_pair[0] + best_pair[1]
    n = len(token_bytes)
    for i in range(n - 1):
        if token_bytes[i] != best_pair[0] or token_bytes[i + 1] != best_pair[1]:
            continue
        if i > 0:
            left = token_bytes[i - 1]
            pair_counts[(left, best_pair[0])] -= count
            pair_counts[(left, merged)] += count
        if i + 2 < n:
            right = token_bytes[i + 2]
            pair_counts[(best_pair[1], right)] -= count
            pair_counts[(merged, right)] += count
        pair_counts[best_pair] -= count


def train_bpe_fast(
    corpus: dict[tuple[str, ...], int],
    vocab_size: int,
    num_special_tokens: int,
) -> list[tuple[bytes, bytes]]:
    merge_num = vocab_size - 256 - num_special_tokens
    merges: list[tuple[bytes, bytes]] = []
    pair_counts = _count_pairs(corpus)

    def tie_key(pc: tuple[tuple[str, str], int]) -> tuple[int, bytes, bytes]:
        p, c = pc
        return (c, gpt2_symbols_to_raw_bytes(p[0]), gpt2_symbols_to_raw_bytes(p[1]))

    for _ in range(merge_num):
        positive = [(p, c) for p, c in pair_counts.items() if c > 0]
        if not positive:
            raise RuntimeError("no pairs left before enough merges")
        best_pair = max(positive, key=tie_key)[0]

        merges.append(
            (
                gpt2_symbols_to_raw_bytes(best_pair[0]),
                gpt2_symbols_to_raw_bytes(best_pair[1]),
            )
        )

        new_corpus: dict[tuple[str, ...], int] = {}
        for token_bytes, count in corpus.items():
            new_tok = merge_token_bytes(token_bytes, best_pair)
            new_corpus[new_tok] = new_corpus.get(new_tok, 0) + count
            update_pair_counts(pair_counts, best_pair, token_bytes, count)
        corpus = new_corpus

    return merges


def generate_vocab(
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> dict[int, bytes]:
    """
    ids: 0 = first special, 1..256 = raw single bytes, 257.. = merge outputs in order.
    """
    u2b = _unicode_to_byte()

    def special_to_bytes(s: str) -> bytes:
        return bytes(u2b[c] for c in s)

    vocab: dict[int, bytes] = {}
    if not special_tokens:
        raise ValueError("expected at least one special token for course tests")
    vocab[0] = special_to_bytes(special_tokens[0])

    for b in range(256):
        vocab[1 + b] = bytes([b])

    for i, (lb, rb) in enumerate(merges):
        vocab[257 + i] = lb + rb

    return vocab


def _build_corpus_from_text(text: str, special_tokens: list[str]) -> dict[tuple[str, ...], int]:
    """Count GPT-2 symbol sequences per pretoken, line by line (matches reference)."""
    corpus: defaultdict[tuple[str, ...], int] = defaultdict(int)
    special_set = set(special_tokens)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for pt in re.findall(GPT2_PAT, line):
            if pt in special_set:
                continue
            key = pretoken_to_symbol_tuple(pt)
            corpus[key] += 1
    return dict(corpus)


def _merge_corpus_dicts(
    partials: list[dict[tuple[str, ...], int]],
) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    for d in partials:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _pretokenize_line_batch(
    args: tuple[tuple[str, ...], tuple[str, ...]],
) -> dict[tuple[str, ...], int]:
    """ProcessPool worker: same counts as `_build_corpus_from_text` on joined lines."""
    lines, special_tokens = args
    if not lines:
        return {}
    chunk = "\n".join(lines)
    return _build_corpus_from_text(chunk, list(special_tokens))


def _build_corpus_from_text_parallel(
    text: str,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> dict[tuple[str, ...], int]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    if n < _MIN_LINES_FOR_PRETOKEN_MP:
        return _build_corpus_from_text(text, special_tokens)

    workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 1))
    workers = min(workers, n)

    # Split line list into contiguous batches (order irrelevant for multiset counts).
    batch_size = (n + workers - 1) // workers
    batches: list[tuple[str, ...]] = []
    for i in range(0, n, batch_size):
        batches.append(tuple(lines[i : i + batch_size]))
    st = tuple(special_tokens)
    worker_args = [(b, st) for b in batches]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        partials = list(pool.map(_pretokenize_line_batch, worker_args))
    return _merge_corpus_dicts(partials)


def bpe_tokenizer_training(
    input_file: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    *,
    pretoken_num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens is None:
        special_tokens = []

    path = Path(input_file)
    text = path.read_text(encoding="utf-8", errors="replace")
    corpus = _build_corpus_from_text_parallel(
        text, special_tokens, num_workers=pretoken_num_workers
    )

    merges = train_bpe_fast(corpus, vocab_size, len(special_tokens))
    vocab = generate_vocab(merges, special_tokens)
    return vocab, merges
