import json
import regex as re

from cs336_basics.bpe_tokenizer_org import bpe_tokenizer_training
from tests.test_tokenizer import VOCAB_PATH, MERGES_PATH
from tests.common import gpt2_bytes_to_unicode

_UNICODE_TO_BYTE: dict[str, int] = {v: k for k, v in gpt2_bytes_to_unicode().items()}
def gpt2_symbols_to_raw_bytes(s: str) -> bytes:
    u2b = _UNICODE_TO_BYTE
    return bytes(u2b[c] for c in s)

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_rank = dict[bytes, int]()
        self.merge_rank = dict[bytes, int]()

        # vocab rank
        self.vocab_rank = {v: k for k, v in vocab.items()}
        # merge rank
        self.merge_rank = {merge: rank for rank, merge in enumerate(merges)}
    
    def _find_chunk_boundaries(
        self,
        corpus: str,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        chunk_size = len(corpus) // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = len(corpus)

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _merge_bytes(self, pre_token: str) -> list[str]:
        pre_token_bytes = pre_token.encode("utf-8")
        pre_token_bytes_list = [bytes([b]) for b in pre_token_bytes]
   
        merged_token_list = list[bytes]()
        # head = pre_token_bytes_list[0]
        # idx = 1
        # while idx < len(pre_token_bytes_list) :
        #     merge_str = head + pre_token_bytes_list[idx]
        #     if merge_str in self.merge_rank:
        #         head = merge_str
        #     else:
        #         merged_token_list.append(self.vocab_rank[head])
        #         head = pre_token_bytes_list[idx]
        #     idx += 1
        # if head == pre_token_bytes_list[-1]:
        #     merged_token_list.append(self.vocab_rank[head])
        while len(pre_token_bytes_list) > 1:
            best_rank = float('inf')
            best_idx = -1
            for i in range(len(pre_token_bytes_list)-1):
                rank = self.merge_rank.get((pre_token_bytes_list[i], pre_token_bytes_list[i+1]), float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = pre_token_bytes_list[best_idx] + pre_token_bytes_list[best_idx+1]
            pre_token_bytes_list = pre_token_bytes_list[:best_idx] + [merged] + pre_token_bytes_list[best_idx+2:]

        merged_token_list = [self.vocab_rank[token] for token in pre_token_bytes_list]
        return merged_token_list


    def _pre_tokenizer_chunk(self, chunk: str, pre_token_list: list[str], special_tokens: list[str]) -> list[str]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        merged_token_list = list[str]()

        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            segments = re.split(pattern, chunk)
        else:
            segments = [chunk]

        for segment in segments:
            if segment in self.special_tokens:
                merged_token_list.append(self.vocab_rank[segment.encode("utf-8")])
            else:
                pre_tokens = re.findall(PAT, segment)
                for pre_token in pre_tokens:
                    merged_token_list.extend(self._merge_bytes(pre_token))

        return merged_token_list

    @classmethod
    def from_files(cls, vocab_filepath: str, merge_filepath: str, special_tokens: list[str] | None = None):
        vocab = dict[int, bytes]()
        merges = list[tuple[bytes, bytes]]()
        vocab_gpt2 = dict[str, int]()

        with open(vocab_filepath, "r") as f:
            vocab_gpt2 = json.load(f)
        for token, id in vocab_gpt2.items():
            vocab[int(id)] = gpt2_symbols_to_raw_bytes(token)
        
        with open(merge_filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                clear_line = line.rstrip()
                if clear_line and len(clear_line.split(" ")) == 2:
                    merges.append(tuple(gpt2_symbols_to_raw_bytes(token) for token in clear_line.split(" ")))

        return cls(vocab, merges, special_tokens)

    def encode(self, corpus: str) -> list[int]:
        pre_token_chunk_size = 512 * 1024  # 512 KB
        chunks_num = max(1, (len(corpus) + pre_token_chunk_size - 1) // pre_token_chunk_size)
   
        merged_token_list = list[str]()

        boundaries = self._find_chunk_boundaries(corpus, chunks_num, b"<|endoftext|>")
        chunk_tasks = [
            (corpus, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        # 1. pre-tokenize & 2. merge
        for chunk_task in chunk_tasks:
            chunk = corpus[chunk_task[1]:chunk_task[2]]
            merged_token_list = self._pre_tokenizer_chunk(chunk, merged_token_list, self.special_tokens)

        return merged_token_list


    # def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = bytearray()
        b2u = gpt2_bytes_to_unicode()
        for id in ids:
            if id in self.vocab:
                decoded_bytes += bytes(self.vocab[id])
           
            else:
                decoded_bytes += bytes("\uFFFD".encode("utf-8"))
           
        return decoded_bytes.decode("utf-8", errors="replace")

# def main():
#     tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])
#     merged_token_list = tokenizer.encode(
# """Once upon a time, there was a girl. She was very small, just three years old.
# The girl had lots of clothes. She wanted to manage them, to keep them tidy. But it was difficult for her, because she was so small.
# One day, the girl found one of her clothes. But it was dead! She cried, because she was sad.
# But the girl was brave. She managed to find all the other clothes, and tidy them up. In the end, everything was nice and tidy.
# The girl was very happy.
# <|endoftext|>
# """)
#     decoded_text = tokenizer.decode(merged_token_list)
#     print(decoded_text)

# if __name__ == "__main__":
#     main()