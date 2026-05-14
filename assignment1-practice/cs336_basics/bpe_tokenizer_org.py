from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import os
from pathlib import Path
import json

from cs336_basics.pretokenization_example_org import find_chunk_boundaries, pre_tokenize_chunk
from collections import defaultdict
from tests.common import gpt2_bytes_to_unicode

# Built once: gpt2_symbols_to_raw_bytes hot path must not rebuild this per call.
_UNICODE_TO_BYTE: dict[str, int] = {v: k for k, v in gpt2_bytes_to_unicode().items()}


def _pretokenize_file_chunk(
    args: tuple[str, int, int, tuple[str, ...]],
) -> dict[tuple[str, ...], int]:
    """Worker: read [start, end) bytes from path, pretokenize, return local counts (picklable)."""
    path, start, end, special_tokens = args
    corpus: dict[tuple[str, ...], int] = {}
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pre_tokenize_chunk(chunk, corpus, list(special_tokens))
    return corpus


def _merge_corpus_parts(parts: list[dict[tuple[str, ...], int]]) -> dict[tuple[str, ...], int]:
    out: dict[tuple[str, ...], int] = {}
    for part in parts:
        for k, v in part.items():
            out[k] = out.get(k, 0) + v
    return out


def _unicode_to_byte() -> dict[str, int]:
    return _UNICODE_TO_BYTE


@lru_cache(maxsize=None)
def gpt2_symbols_to_raw_bytes(s: str) -> bytes:
    u2b = _UNICODE_TO_BYTE
    return bytes(u2b[c] for c in s)


def tie_key(pc: tuple[tuple[str, str], int]) -> tuple[int, bytes, bytes]:
    """Tie-break: highest count, then lexicographically greatest (lb, rb) as raw bytes."""
    p, c = pc
    return (c, gpt2_symbols_to_raw_bytes(p[0]), gpt2_symbols_to_raw_bytes(p[1]))


def merge_token_bytes(
    token_bytes: tuple[str, ...], best_pair: tuple[str, str]
) -> tuple[str, ...]:
    a, b = best_pair
    merged = a + b
    out: list[str] = []
    i = 0
    n = len(token_bytes)
    while i < n:
        if i + 1 < n and token_bytes[i] == a and token_bytes[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(token_bytes[i])
            i += 1
    return tuple(out)

# def update_pair_counts(pair_counts: dict[tuple, int], best_pair: tuple[int, int], token_bytes: tuple[int, int], new_token_bytes: tuple[int, int], heap: list[tuple[int, tuple[int, int]]], count: int):
def update_pair_counts(
    pair_counts: defaultdict[tuple[str, str], int],
    best_pair: tuple[str, str],
    token_bytes: tuple[str, ...],
    count: int,
) -> None:
    merged = best_pair[0] + best_pair[1]
    for i in range(0, len(token_bytes) - 1):
        if token_bytes[i] == best_pair[0] and token_bytes[i + 1] == best_pair[1]:                                                                                                                                                          
            if i > 0:                                   
                l_token_byte = token_bytes[i-1]                                                                                                                                                                                                
                pair_counts[(l_token_byte, best_pair[0])] -= count  
                pair_counts[(l_token_byte, merged)] += count        
                # new_pair = (l_token_byte, merged)                                                                                                                
                # heapq.heappush(heap, (-pair_counts[new_pair], new_pair))                                                                                                                                     
            if i + 2 < len(token_bytes):                                                                                                                                              
                r_token_byte = token_bytes[i+2]                                                                                                                                                                                                
                pair_counts[(best_pair[1], r_token_byte)] -= count                                                                                                                       
                pair_counts[(merged, r_token_byte)] += count
                # new_pair = (merged, r_token_byte)                                                                                                                      
                # heapq.heappush(heap, (-pair_counts[new_pair], new_pair))                                                                                                                                     
            pair_counts[best_pair] -= count


def train_bpe_fast(
    corpus: dict[tuple, int], vocab_size: int, num_special_tokens: int = 0
):
    merge_num = vocab_size - 256 - num_special_tokens
    pair_counts = defaultdict(int)
    merges: list[tuple[bytes, bytes]] = []

    for token_bytes, count in corpus.items():
        for i in range(len(token_bytes) - 1):
            pair_counts[(token_bytes[i], token_bytes[i + 1])] += count

    # heap = [(-count, pair) for pair, count in pair_counts.items()]
    # heapq.heapify(heap)

    merge_idx = 0
    for _ in range(merge_num):
        # fix: lazy deletion
        # while heap:
        #     neg_count, best_pair = heapq.heappop(heap)
        #     if pair_counts.get(best_pair, 0) == -neg_count:
        #         break

        # merges.append((best_pair[0], best_pair[1]))
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

        new_corpus = {}
        for token_bytes, count in corpus.items():
            new_token_bytes = merge_token_bytes(token_bytes, best_pair)
            new_corpus[new_token_bytes] = new_corpus.get(new_token_bytes, 0) + count
            # update_pair_counts(pair_counts, best_pair, token_bytes, new_token_bytes, heap, count)
            update_pair_counts(pair_counts, best_pair, token_bytes, count)
        corpus = new_corpus

    return merges



def bytes_to_unicode() -> dict[int, str]:
      bs = (list(range(ord("!"), ord("~") + 1)) +                                                                                                                                                                                                
            list(range(ord("¡"), ord("¬") + 1)) +                                                                                                                                                                                                
            list(range(ord("®"), ord("ÿ") + 1)))
      cs = bs[:]                                                                                                                                                                                                                                 
      n = 0       
      for b in range(256):                                                                                                                                                                                                                       
          if b not in bs:
              bs.append(b)                                                                                                                                                                                                                       
              cs.append(256 + n)
              n += 1
      return dict(zip(bs, [chr(c) for c in cs]))

def generate_vocab(
    merges: list[tuple[bytes, bytes]], special_tokens: list[str]
) -> dict[str, bytes]:
    vocab = dict[str, bytes]()
    b2u = gpt2_bytes_to_unicode()

    # Build reverse: iterate in the ORDER bytes_to_unicode built them                                                                                                                                                                          
      # i.e. the same bs list order
    bs = (list(range(ord("!"), ord("~") + 1)) +                                                                                                                                                                                                
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1)))                                                                                                                                                                                                 
    # add remaining bytes in order they were appended                                                                                                                                                                                          
    for b in range(256):                                                                                                                                                                                                                       
        if b not in bs:                                                                                                                                                                                                                        
            bs.append(b)                                                                                                                                                                                                                       
                
    vocab: dict[str, int] = {}

    # for i in range(len(special_tokens)):
    #     vocab[special_tokens[i]] = i
    # for i, byte_val in enumerate(bs):        # ← iterate bs order, not range(256)
    #     vocab[b2u[byte_val]] = i + len(special_tokens)
    # for i in range(len(merges)):
    #     merged_str = merges[i][0] + merges[i][1]
    #     vocab[merged_str] = i + len(special_tokens) + 256
    def special_to_bytes(s: str) -> bytes:
        return bytes(_UNICODE_TO_BYTE[c] for c in s)
    if not special_tokens:
        raise ValueError("expected at least one special token for course tests")
    # Add all special tokens first
    for i, special in enumerate(special_tokens):
        vocab[i] = special_to_bytes(special)

    # Add byte tokens, with indices offset by number of special tokens
    offset = len(special_tokens)
    for b in range(256):
        vocab[offset + b] = bytes([b])

    # Add merge tokens, with indices after special and byte tokens
    for i, (lb, rb) in enumerate(merges):
        vocab[offset + 256 + i] = lb + rb

   
    return vocab


def bpe_tokenizer_training(
    input_file: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
):
    if special_tokens is None:
        special_tokens = []
    if num_processes is None:
        num_processes = min(4, os.cpu_count() or 1)

    path_resolved = str(Path(input_file).resolve())
    st_tuple = tuple(special_tokens)

    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_tasks = [
        (path_resolved, start, end, st_tuple)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    if num_processes <= 1 or len(chunk_tasks) <= 1:
        corpus = _merge_corpus_parts([_pretokenize_file_chunk(t) for t in chunk_tasks])
    else:
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            parts = list(pool.map(_pretokenize_file_chunk, chunk_tasks))
        corpus = _merge_corpus_parts(parts)

    b2u = gpt2_bytes_to_unicode()

    # Convert corpus keys from raw chars to GPT-2 unicode
    converted_corpus: dict[tuple[str, ...], int] = {}
    for token_bytes, count in corpus.items():
        new_key = tuple(
            b2u[byte]
            for char in token_bytes
            for byte in char.encode("utf-8")
        )
        converted_corpus[new_key] = converted_corpus.get(new_key, 0) + count

    merges = train_bpe_fast(converted_corpus, vocab_size, len(special_tokens))
    vocab = generate_vocab(merges, special_tokens)

    # with open("vocab.json", "w") as f:
    #     json.dump(vocab, f)
    # with open("merges.txt", "w") as f:
    #     for merge in merges:
    #         f.write(f"{merge[0]} {merge[1]}\n")

    return vocab, merges

def main():
    vocab, merges = bpe_tokenizer_training("data/TinyStoriesV2-GPT4-valid.txt", 300, ["<|endoftext|>"])

    return vocab, merges

if __name__ == "__main__":
    main()
