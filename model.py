import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Iterable


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

  def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[Dict] = None):
    q = self.query(x)

    if kv_cache is None or xa is None or self.key not in kv_cache:
      k, v = self.key(x if xa is None else xa), self.value(x if xa is None else xa)
    else:
      k, v = kv_cache[self.key], kv_cache[self.value]

    wv, qk = self.qkv_attention(q, k, v, mask)
    return self.out(wv), qk

  def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    qk = (q * scale) @ (k * scale).transpose(-1, -2)
    if mask is not None:
      qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
    qk = qk.detach()

    return out, qk


class ResidualAttentionBlock(nn.Module):
  def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
    super().__init__()
    self.attn = MultiHeadAttention(n_state, n_head)
    self.attn_ln = nn.LayerNorm(n_state)

    self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
    self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

    self.mlp = nn.Sequential(nn.Linear(n_state, n_state * 4), nn.GELU(), nn.Linear(n_state * 4, n_state))
    self.mlp_ln = nn.LayerNorm(n_state)

  def forward(self, x: Tensor, xa: Optional[Tensor] = None, mask: Optional[Tensor] = None, kv_cache: Optional[Dict] = None):
    x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
    if self.cross_attn:
      x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
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

  def forward(self):
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
    self.token_embedding = nn.Embedding(n_vocab, n_text_state)
    self.positional_embedding = nn.Parameter(torch.empty(n_text_ctx, n_text_state))

    self.blocks = nn.Sequential(*[ResidualAttentionBlock(n_text_state, n_text_head, cross_attention=True) for _ in range(n_text_layer)])
    self.ln = nn.LayerNorm(n_text_state)
    mask = torch.empty(n_text_ctx, n_text_ctx).fill_(-np.inf).triu(1)
    self.register_buffer('mask', mask, persistent=False)

  def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[Dict] = None):
    offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
    x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
    x = self.blocks(x, xa, self.mask, kv_cache=kv_cache)
    x = self.ln(x)
    x = x @ self.token_embedding.weight.T
    return x


class Model(nn.Module):
  def __init__(self, dims):
    super().__init__()
    self.encoder = AudioEncoder(**dims)
    self.decoder = TextDecoder(**dims)

  def forward(self):
    pass
