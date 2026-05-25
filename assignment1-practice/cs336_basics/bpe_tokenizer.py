"""
Byte-level BPE training aligned with GPT-2 / course reference tests.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import regex as re

from tests.common import gpt2_bytes_to_unicode

_MIN_LINES_FOR_PRETOKEN_MP = 1000
_LINE_BATCH_SIZE = 8192

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


def _bytes_to_gpt2_display(b: bytes) -> str:
    b2u = gpt2_bytes_to_unicode()
    return "".join(b2u[x] for x in b)


def save_vocab_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """Save vocab/merges in GPT-2-style JSON + merges.txt (for ``Tokenizer.from_files``)."""
    vocab_json = {_bytes_to_gpt2_display(token): idx for idx, token in vocab.items()}
    Path(vocab_path).write_text(json.dumps(vocab_json), encoding="utf-8")
    merge_lines = [
        f"{_bytes_to_gpt2_display(left)} {_bytes_to_gpt2_display(right)}" for left, right in merges
    ]
    Path(merges_path).write_text("\n".join(merge_lines) + ("\n" if merge_lines else ""), encoding="utf-8")


class BPETokenizer:
    """Encode text with a trained BPE vocab + merge list from ``bpe_tokenizer_training``."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens or [])
        self.vocab_rank = {token: idx for idx, token in vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self._special_ids = {
            tok: self.vocab_rank[gpt2_symbols_to_raw_bytes(tok)] for tok in self.special_tokens
        }
        if not self.special_tokens:
            raise ValueError("expected at least one special token")
        self.eot_id = self._special_ids[self.special_tokens[0]]

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        from cs336_basics.tokenizer import Tokenizer

        tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
        return cls(tok.vocab, tok.merges, tok.special_tokens)

    def _merge_pre_token(self, pre_token: str) -> list[int]:
        pre_token_bytes_list = [bytes([b]) for b in pre_token.encode("utf-8")]
        while len(pre_token_bytes_list) > 1:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(pre_token_bytes_list) - 1):
                pair = (pre_token_bytes_list[i], pre_token_bytes_list[i + 1])
                rank = self.merge_rank.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = pre_token_bytes_list[best_idx] + pre_token_bytes_list[best_idx + 1]
            pre_token_bytes_list = (
                pre_token_bytes_list[:best_idx] + [merged] + pre_token_bytes_list[best_idx + 2 :]
            )
        return [self.vocab_rank[token] for token in pre_token_bytes_list]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            segments = re.split(pattern, text)
        else:
            segments = [text]

        for segment in segments:
            if segment in self._special_ids:
                ids.append(self._special_ids[segment])
            elif segment:
                for pre_token in re.findall(GPT2_PAT, segment):
                    if pre_token in self._special_ids:
                        ids.append(self._special_ids[pre_token])
                    else:
                        ids.extend(self._merge_pre_token(pre_token))
        return ids

    def encode_line(self, line: str, *, append_eot: bool = True) -> list[int]:
        line = line.strip()
        if not line:
            return [self.eot_id] if append_eot else []
        ids = self.encode(line)
        if append_eot:
            ids.append(self.eot_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = bytearray()
        for token_id in ids:
            if token_id in self.vocab:
                decoded_bytes.extend(self.vocab[token_id])
            else:
                decoded_bytes.extend("\ufffd".encode("utf-8"))
        return decoded_bytes.decode("utf-8", errors="replace")


_MP_TOKENIZER: BPETokenizer | None = None


def _tokenize_worker_init(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    global _MP_TOKENIZER
    _MP_TOKENIZER = BPETokenizer(vocab, merges, special_tokens)


def _encode_lines_batch(lines: list[str]) -> list[list[int]]:
    assert _MP_TOKENIZER is not None
    return [_MP_TOKENIZER.encode_line(line) for line in lines]


def _count_lines_batch(lines: list[str]) -> int:
    assert _MP_TOKENIZER is not None
    return sum(len(_MP_TOKENIZER.encode_line(line)) for line in lines)


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is not None:
        return max(1, num_workers)
    return max(1, (os.cpu_count() or 1) - 1)


def _split_for_workers(lines: list[str], workers: int) -> list[list[str]]:
    if not lines:
        return []
    workers = min(workers, len(lines))
    chunk = (len(lines) + workers - 1) // workers
    return [lines[i : i + chunk] for i in range(0, len(lines), chunk)]


def _iter_line_batches(path: Path, batch_size: int):
    batch: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def count_tokens_in_text_file(
    txt_path: str | os.PathLike,
    tokenizer: BPETokenizer,
    num_workers: int | None = None,
) -> int:
    txt_path = Path(txt_path)
    workers = _resolve_num_workers(num_workers)
    line_batches = list(_iter_line_batches(txt_path, _LINE_BATCH_SIZE))
    if workers <= 1:
        return sum(len(tokenizer.encode_line(line)) for batch in line_batches for line in batch)

    initargs = (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)
    total = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_tokenize_worker_init, initargs=initargs) as pool:
        for batch in line_batches:
            for n in pool.map(_count_lines_batch, _split_for_workers(batch, workers)):
                total += n
    return total


def tokenize_text_file_to_memmap(
    txt_path: str | os.PathLike,
    out_path: str | os.PathLike,
    tokenizer: BPETokenizer,
    num_workers: int | None = None,
) -> int:
    """Write uint16 memmap tokens for ``training_loop.load_memmap``."""
    txt_path = Path(txt_path)
    out_path = Path(out_path)
    workers = _resolve_num_workers(num_workers)
    if workers > 1:
        print(f"  parallel tokenize with {workers} processes")

    total = count_tokens_in_text_file(txt_path, tokenizer, num_workers=num_workers)
    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total,))
    idx = 0
    line_batches = list(_iter_line_batches(txt_path, _LINE_BATCH_SIZE))

    if workers <= 1:
        for batch in line_batches:
            for line in batch:
                ids = tokenizer.encode_line(line)
                n = len(ids)
                if n:
                    arr[idx : idx + n] = ids
                    idx += n
    else:
        initargs = (tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)
        with ProcessPoolExecutor(max_workers=workers, initializer=_tokenize_worker_init, initargs=initargs) as pool:
            for batch in line_batches:
                for encoded in pool.map(_encode_lines_batch, _split_for_workers(batch, workers)):
                    for ids in encoded:
                        n = len(ids)
                        if n:
                            arr[idx : idx + n] = ids
                            idx += n

    arr.flush()
    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Train BPE (bpe_tokenizer.py) and write uint16 memmaps.")
    p.add_argument("--train-txt", type=Path, required=True, help="Text file to train BPE on")
    p.add_argument("--corpus-txt", type=Path, action="append", help="Text file(s) to tokenize (default: train-txt)")
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    p.add_argument("--vocab-out", type=Path, default=None, help="defaults to out-dir/vocab.json")
    p.add_argument("--merges-out", type=Path, default=None, help="defaults to out-dir/merges.txt")
    p.add_argument("--skip-train", action="store_true", help="load existing vocab/merges instead of training")
    p.add_argument("--pretoken-num-workers", type=int, default=None, help="CPU processes for BPE pretokenize")
    p.add_argument("--tokenize-num-workers", type=int, default=None, help="CPU processes for .txt -> .bin encode")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.vocab_out or (args.out_dir / "vocab.json")
    merges_path = args.merges_out or (args.out_dir / "merges.txt")

    if args.skip_train:
        if not vocab_path.is_file() or not merges_path.is_file():
            raise FileNotFoundError(
                f"missing {vocab_path} or {merges_path}. "
                "Run without --skip-train first to train BPE and create them."
            )
        tokenizer = BPETokenizer.from_files(vocab_path, merges_path, args.special_token)
        print(f"loaded tokenizer from {vocab_path} ({len(tokenizer.vocab)} tokens)")
    else:
        print(f"training BPE on {args.train_txt} (vocab_size={args.vocab_size}) ...")
        vocab, merges = bpe_tokenizer_training(
            str(args.train_txt),
            args.vocab_size,
            args.special_token,
            pretoken_num_workers=args.pretoken_num_workers,
        )
        save_vocab_merges(vocab, merges, vocab_path, merges_path)
        tokenizer = BPETokenizer(vocab, merges, args.special_token)
        print(f"saved {vocab_path} and {merges_path} ({len(vocab)} tokens)")

    corpus_files = args.corpus_txt or [args.train_txt]
    for txt in corpus_files:
        out = args.out_dir / f"{txt.stem}.bin"
        print(f"tokenizing {txt} -> {out} ...")
        n = tokenize_text_file_to_memmap(txt, out, tokenizer, num_workers=args.tokenize_num_workers)
        print(f"  {n:,} tokens")


if __name__ == "__main__":
    main()
