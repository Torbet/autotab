import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def sinusoids(length, channels, max_timescale=10000):
  assert channels % 2 == 0
  log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
  inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
  scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
  return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
  def __init__(self, n_state: int, n_head: int):
    super().__init__()
    self.n_head = n_head
    self.query = nn.Linear(n_state, n_state)
    self.key = nn.Linear(n_state, n_state, bias=False)
    self.value = nn.Linear(n_state, n_state)
    self.out = nn.Linear(n_state, n_state)

  def forward(self, x: torch.Tensor, xa: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    q = self.query(x)
    k = self.key(x if xa is None else xa)
    v = self.value(x if xa is None else xa)

    wv, qk = self.qkv_attention(q, k, v, mask)
    return self.out(wv), qk

  def qkv_attention(
    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    qk = torch.matmul(q * scale, (k * scale).transpose(-1, -2))
    if mask is not None:
      qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    out = torch.matmul(w, v).permute(0, 2, 1, 3).flatten(start_dim=2)

    return out, qk.detach()


class ResidualAttentionBlock(nn.Module):
  def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
    super().__init__()
    self.attn = MultiHeadAttention(n_state, n_head)
    self.attn_ln = nn.LayerNorm(n_state)

    self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
    self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

    self.mlp = nn.Sequential(nn.Linear(n_state, n_state * 4), nn.GELU(), nn.Linear(n_state * 4, n_state))
    self.mlp_ln = nn.LayerNorm(n_state)

  def forward(
    self,
    x: torch.Tensor,
    xa: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    x = x + self.attn(self.attn_ln(x), mask=mask)[0]
    if self.cross_attn:
      x = x + self.cross_attn(self.cross_attn_ln(x), xa)[0]
    x = x + self.mlp(self.mlp_ln(x))
    return x


class AudioEncoder(nn.Module):
  def __init__(self, n_mels: int, n_audio_ctx: int, n_audio_state: int, n_audio_head: int, n_audio_layer: int, **kwargs):
    super().__init__()
    self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, padding=1)
    self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
    self.register_buffer('positional_embedding', sinusoids(n_audio_ctx, n_audio_state))

    self.blocks = nn.Sequential(*[ResidualAttentionBlock(n_audio_state, n_audio_head) for _ in range(n_audio_layer)])

    self.ln_post = nn.LayerNorm(n_audio_state)

  def forward(self, x: Tensor):
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)
    assert x.shape[1:] == self.positional_embedding.shape, 'incorrect audio shape'
    x = x + self.positional_embedding
    x = self.blocks(x)
    x = self.ln_post(x)
    return x


class TextDecoder(nn.Module):
  def __init__(self, n_vocab: int, n_text_ctx: int, n_text_state: int, n_text_head: int, n_text_layer: int, **kwargs):
    super().__init__()
    self.n_text_ctx = n_text_ctx
    self.max_tokens_to_sample = n_text_ctx // 2
    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = nn.Parameter(torch.empty(n_text_ctx, n_text_state))

    self.blocks = nn.ModuleList([ResidualAttentionBlock(n_text_state, n_text_head, cross_attention=True) for _ in range(n_text_layer)])
    self.ln = nn.LayerNorm(n_text_state)
    mask = torch.empty(n_text_ctx, n_text_ctx).fill_(-np.inf).triu(1)
    self.register_buffer('mask', mask, persistent=False)

  def forward(self, x: Tensor, xa: Tensor):
    seq_len = min(x.shape[-1], self.n_text_ctx)
    x = x[:, :seq_len]

    token_emb = self.token_embedding(x)
    pos_emb = self.positional_embedding[:seq_len]

    x = token_emb + pos_emb

    for block in self.blocks:
      x = block(x, xa, mask=self.mask[:seq_len, :seq_len])

    x = self.ln(x)
    x = x @ self.token_embedding.weight.T
    return x


class Transformer(nn.Module):
  def __init__(self, dims):
    super().__init__()
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)

  def forward(self, mel: Tensor, tokens: Tensor, teacher_forcing: bool = False, teacher_forcing_rate: float = 1.0):
    enc = self.encoder(mel)
    if not teacher_forcing:
      return self.decoder(tokens, enc)
    else:
      BS, T = tokens.shape
      out = torch.full((BS, 1), 134, dtype=torch.long, device=tokens.device)
      for i in range(T - 1):
        if torch.rand(1) < teacher_forcing_rate:
          next_token = tokens[:, i + 1].unsqueeze(1)
        else:
          next_token = self.decoder(out, enc)[:, -1].argmax(1).unsqueeze(1)
        out = torch.cat([out, next_token], dim=1)
      return self.decoder(out, enc)
