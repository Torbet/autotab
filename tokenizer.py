import tiktoken


def create_tab_tokenizer():
  tokens = []
  tokens.extend([f'<S{i}>' for i in range(1, 7)])
  tokens.extend([f'<F{i}>' for i in range(1, 25)])
  tokens.extend([f'<T{2**i}>' for i in range(6)])
  tokens.extend(['<H>', '<P>', '<S>', '<B>'])
  special = ['<|endoftext|>', '<|startoftab|>', '<|endoftab|>']
  special.extend([f'<U{i}>' for i in range(51861 - len(tokens))])
  ranks = {token.encode(): i for i, token in enumerate(tokens)}
  special = {token: len(ranks) + i for i, token in enumerate(special)}
  n_vocab = len(ranks) + len(special)
  return tiktoken.Encoding(name='tab', explicit_n_vocab=n_vocab, pat_str=r'<[^>]+>|[^\s<]+|\s+', mergeable_ranks=ranks, special_tokens=special)
