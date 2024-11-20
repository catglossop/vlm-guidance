"""Jax implementation of RT-1 / RT-1-X (https://arxiv.org/abs/2212.06817).

This is based on the tensorflow implementation in:
https://github.com/google-research/robotics_transformer and also includes
improvements made in RT-X (https://arxiv.org/abs/2310.08864).
"""

import enum
from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import efficientnet
import film_conditioning

class FFNOptions(enum.Enum):
  """Different choices of FFN block for ablation testing."""

  LINEAR = 'linear'  # RT-1 Legacy
  SWIGLU = 'swiglu'  # Match LLaMa


class TransformerBlock(nn.Module):
  """A self-attention transformer block.

  See the `_TransformerLayer` in
  google-research/robotics_transformer/transformer.py for the original
  tensorflow implementation.
  """
  layer_size: int = 128
  num_heads: int = 8
  feed_forward_hidden_size: int = 512
  feed_forward_output_size: int = 512
  ffn_option: FFNOptions = FFNOptions.SWIGLU
  dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray, *, train: bool):
    x1 = nn.LayerNorm()(x)

    x1 = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_features=(self.layer_size * self.num_heads),
        dropout_rate=self.dropout_rate,
    )(x1, x1, mask=attn_mask, deterministic=not train)

    x = x + x1

    y = nn.LayerNorm()(x)

    if self.ffn_option == FFNOptions.SWIGLU:
      h1 = nn.Dense(self.feed_forward_hidden_size, use_bias=False)(y)
      h1 = nn.swish(h1)
      gate = nn.Dense(self.feed_forward_hidden_size, use_bias=False)(y)
      ff_y = nn.Dense(self.feed_forward_output_size, use_bias=False)(h1 * gate)
    elif self.ffn_option == FFNOptions.LINEAR:
      ff_y = nn.Dense(self.feed_forward_output_size, use_bias=False)(y)
    else:
      raise ValueError(f'Unknown FFN option: {self.ffn_option}')

    ff_y = nn.Dropout(self.dropout_rate)(ff_y, deterministic=not train)
    x = x + ff_y
    return x


class Transformer(nn.Module):
  """Transformer architecture with dense positional embedding.

  See the `Transformer` in
  google-research/robotics_transformer/transformer.py for the original
  tensorflow implementation.
  """

  num_layers: int = 8
  layer_size: int = 128
  num_heads: int = 8
  feed_forward_hidden_size: int = 512
  feed_forward_output_size: int = 512
  ffn_option: FFNOptions = FFNOptions.SWIGLU
  dropout_rate: float = 0.1
  vocab_size: int = 256

  @nn.compact
  def __call__(self, x: jnp.ndarray, attn_mask: jnp.ndarray, *, train: bool):
    bs, seqlen, *_ = x.shape

    pos = jnp.expand_dims(jnp.arange(0, seqlen, 1), 0)
    pos = jnp.tile(pos, [bs, 1])
    pos = jax.nn.one_hot(pos, seqlen)

    x = nn.Dense(self.feed_forward_output_size)(x)
    pos_emb = nn.Dense(self.feed_forward_output_size)(pos)
    x += pos_emb

    for _ in range(self.num_layers):
      x = TransformerBlock(
          layer_size=self.layer_size,
          num_heads=self.num_heads,
          feed_forward_hidden_size=self.feed_forward_hidden_size,
          feed_forward_output_size=self.feed_forward_output_size,
          dropout_rate=self.dropout_rate,
          ffn_option=self.ffn_option,
      )(x, attn_mask, train=train)

    output_tokens = nn.Dense(self.vocab_size)(x)
    return output_tokens