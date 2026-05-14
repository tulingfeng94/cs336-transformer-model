  from your_bpe_file import train_bpe_fast, generate_vocab
  from pretokenization_example import find_chunk_boundaries, pre_tokenize_chunk                                                                                                                                                                  
  from collections import defaultdict                                                                                                                                                                                                            
                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                 
  def run_train_bpe(
      input_path: str,
      vocab_size: int,                                                                                                                                                                                                                           
      special_tokens: list[str],
  ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:                                                                                                                                                                                       
                  
      return bpe_tokenizer_training(input_path, vocab_size, special_tokens)