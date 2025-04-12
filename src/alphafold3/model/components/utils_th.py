import torch
import numpy as np
from typing import Dict, Any, Optional, Iterable
import contextlib
import numbers
from collections import abc

VALID_DTYPES = [torch.float32, torch.float64, torch.int8, torch.int32, torch.int64, torch.bool]


def remove_invalidly_typed_feats(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove features of types we don't want to send to the TPU e.g. strings."""
    return {
        k: v
        for k, v in batch.items()
        if isinstance(v, torch.Tensor) and v.dtype in VALID_DTYPES
    }


def bfloat16_getter(next_getter, value, context):
    """Ensures that a bfloat16 parameter is provided by casting if necessary."""
    if context.original_dtype == torch.bfloat16:
        if value.dtype != torch.bfloat16:
            value = value.to(torch.bfloat16)
    return next_getter(value)


@contextlib.contextmanager
def bfloat16_context():
    """Context manager for bfloat16 custom getter."""
    # PyTorch does not have a direct equivalent of Haiku's custom_getter,
    # so we need to handle bfloat16 casting manually.
    try:
        yield
    finally:
        pass


def mask_mean(mask, value, axis=None, keepdims=False, eps=1e-10):
  """Masked mean."""

  mask_shape = mask.shape
  value_shape = value.shape

  assert len(mask_shape) == len(
      value_shape
  ), 'Shapes are not compatible, shapes: {}, {}'.format(mask_shape, value_shape)

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(
      axis, abc.Iterable
  ), 'axis needs to be either an iterable, integer or "None"'

  broadcast_factor = 1.0
  for axis_ in axis:
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      error = f'Shapes are not compatible, shapes: {mask_shape}, {value_shape}'
      assert mask_size == value_size, error

#   return jnp.sum(mask * value, keepdims=keepdims, axis=axis) / (
#       jnp.maximum(
#           jnp.sum(mask, keepdims=keepdims, axis=axis) * broadcast_factor, eps
#       )
#   )
  sum_mask_value = torch.sum(mask * value, dim=axis, keepdim=keepdims)
    
  # 计算 mask 的和，并乘以 broadcast_factor
  sum_mask = torch.sum(mask, dim=axis, keepdim=keepdims) * broadcast_factor
  
  # 计算最大值与 eps 之间的比较
  denominator = torch.maximum(sum_mask, torch.full_like(sum_mask, eps))  # 修复部分
  
  # 返回最终结果
  return sum_mask_value / denominator