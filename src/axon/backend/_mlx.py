"""MLX 백엔드. lazy graph + 자동 async_eval — user 는 NumPy 와 동일 API."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import mlx.core as mx
import numpy as np

from axon.backend._dtype import to_backend_dtype
from axon.backend.protocol import RandomProtocol
from axon.dtype import DType


def _ae[T](fn: Callable[..., T]) -> Callable[..., T]:
  """반환된 mx.array (또는 mx.array list) 에 자동 async_eval 적용."""

  @wraps(fn)
  def wrapped(*args: Any, **kwargs: Any) -> T:
    result = fn(*args, **kwargs)
    if isinstance(result, mx.array):
      mx.async_eval(result)
    elif isinstance(result, list) and result and isinstance(result[0], mx.array):
      mx.async_eval(*result)
    return result

  return wrapped


class _MLXRandom:
  def normal(
    self,
    shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    return mx.random.normal(
      shape=shape, loc=mean, scale=std, dtype=to_backend_dtype(dtype, "mlx")
    )

  def uniform(
    self,
    shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    return mx.random.uniform(
      low=low, high=high, shape=shape, dtype=to_backend_dtype(dtype, "mlx")
    )

  def bernoulli(
    self,
    p: float,
    shape: tuple[int, ...],
    dtype: type[DType] = DType.FLOAT32,
  ):
    return mx.random.bernoulli(p=p, shape=shape).astype(to_backend_dtype(dtype, "mlx"))

  def seed(self, s: int) -> None:
    mx.random.seed(s)


class MLXBackend:
  name: str = "mlx"
  random: RandomProtocol

  def __init__(self) -> None:
    self.random = cast(RandomProtocol, _MLXRandom())

  def array(self, data: Any, dtype: type[DType] = DType.FLOAT32):
    return mx.array(data, dtype=to_backend_dtype(dtype, "mlx"))

  def zeros(self, shape, dtype: type[DType] = DType.FLOAT32):
    return mx.zeros(shape, dtype=to_backend_dtype(dtype, "mlx"))

  def ones(self, shape, dtype: type[DType] = DType.FLOAT32):
    return mx.ones(shape, dtype=to_backend_dtype(dtype, "mlx"))

  def full(self, shape, fill_value, dtype: type[DType] = DType.FLOAT32):
    return mx.full(shape, fill_value, dtype=to_backend_dtype(dtype, "mlx"))

  def zeros_like(self, x):
    return mx.zeros_like(x)

  def ones_like(self, x):
    return mx.ones_like(x)

  def full_like(self, x, fill_value: float):
    return mx.full(x.shape, fill_value, dtype=x.dtype)

  def eye(self, n: int, dtype: type[DType] = DType.FLOAT32):
    return mx.eye(n, dtype=to_backend_dtype(dtype, "mlx"))

  def arange(self, start, stop=None, step=1.0, dtype: type[DType] = DType.FLOAT32):
    if stop is None:
      return mx.arange(start, dtype=to_backend_dtype(dtype, "mlx"))
    return mx.arange(start, stop, step, dtype=to_backend_dtype(dtype, "mlx"))

  def exp(self, x):
    return mx.exp(x)

  def log(self, x):
    return mx.log(x)

  def sqrt(self, x):
    return mx.sqrt(x)

  def rsqrt(self, x):
    return mx.rsqrt(x)

  def abs(self, x):
    return mx.abs(x)

  def sign(self, x):
    return mx.sign(x)

  def maximum(self, a, b):
    return mx.maximum(a, b)

  def minimum(self, a, b):
    return mx.minimum(a, b)

  def clip(self, x, a_min, a_max):
    return mx.clip(x, a_min, a_max)

  def sin(self, x):
    return mx.sin(x)

  def cos(self, x):
    return mx.cos(x)

  def tanh(self, x):
    return mx.tanh(x)

  def power(self, a, b):
    return mx.power(a, b)

  def sum(self, x, axis=None, keepdims=False):
    return mx.sum(x, axis=axis, keepdims=keepdims)

  def mean(self, x, axis=None, keepdims=False):
    return mx.mean(x, axis=axis, keepdims=keepdims)

  def var(self, x, axis=None, keepdims=False, ddof=0):
    return mx.var(x, axis=axis, keepdims=keepdims, ddof=ddof)

  def max(self, x, axis=None, keepdims=False):
    return mx.max(x, axis=axis, keepdims=keepdims)

  def min(self, x, axis=None, keepdims=False):
    return mx.min(x, axis=axis, keepdims=keepdims)

  def norm(self, x, ord=None, axis=None, keepdims=False):
    return mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

  def reshape(self, x, shape):
    return mx.reshape(x, shape)

  def transpose(self, x, axes=None):
    return mx.transpose(x, axes)

  def expand_dims(self, x, axis):
    return mx.expand_dims(x, axis)

  def squeeze(self, x, axis=None):
    return mx.squeeze(x, axis)

  def broadcast_to(self, x, shape):
    return mx.broadcast_to(x, shape)

  def concatenate(self, arrays, axis=0):
    return mx.concatenate(arrays, axis=axis)

  def stack(self, arrays, axis=0):
    return mx.stack(arrays, axis=axis)

  def split(self, x, indices_or_sections, axis=0):
    return list(mx.split(x, indices_or_sections, axis=axis))

  def flip(self, x, axis=None):
    # mlx 에 mx.flip 이 없어 슬라이싱으로 구현.
    if axis is None:
      axes = tuple(range(x.ndim))
    elif isinstance(axis, int):
      axes = (axis,)
    else:
      axes = tuple(axis)
    slicer = [slice(None)] * x.ndim
    for ax in axes:
      slicer[ax] = slice(None, None, -1)
    return x[tuple(slicer)]

  def matmul(self, a, b):
    return mx.matmul(a, b)

  def einsum(self, subscripts, *operands):
    return mx.einsum(subscripts, *operands)

  def where(self, condition, x, y):
    return mx.where(condition, x, y)

  def take(self, x, indices, axis=None):
    return mx.take(x, indices, axis=axis)

  def tril(self, x, k=0):
    return mx.tril(x, k=k)

  def triu(self, x, k=0):
    return mx.triu(x, k=k)

  def sort(self, x, axis=-1):
    return mx.sort(x, axis=axis)

  def argsort(self, x, axis=-1):
    return mx.argsort(x, axis=axis)

  def argmax(self, x, axis=None, keepdims=False):
    # mx.argmax 는 uint32 반환 — numpy 의 int64 와 dtype 다름 (값은 동일).
    if axis is None:
      return mx.argmax(x, keepdims=keepdims)
    return mx.argmax(x, axis=axis, keepdims=keepdims)

  def argmin(self, x, axis=None, keepdims=False):
    if axis is None:
      return mx.argmin(x, keepdims=keepdims)
    return mx.argmin(x, axis=axis, keepdims=keepdims)

  def cumsum(self, x, axis=None):
    return mx.cumsum(x, axis=axis)

  def pad(self, x, pad_width, constant_values=0.0):
    return mx.pad(x, pad_width, constant_values=constant_values)

  def to_numpy(self, x):
    mx.eval(x)
    return np.array(x)

  def from_numpy(self, x):
    return mx.array(np.asarray(x))

  def eval(self, *arrays):
    mx.eval(*arrays)

  def async_eval(self, *arrays):
    mx.async_eval(*arrays)


# graph boundary (reduction / argmax) 에만 async_eval — 모든 op 에 걸면 fusion
# 기회를 잃어 NumPy 보다 느려짐. 끝 op 에서만 chain 전체가 한 번에 evaluate.
_AUTO_AE_METHODS = {
  "sum",
  "mean",
  "var",
  "max",
  "min",
  "norm",
  "argmax",
  "argmin",
}


def _wrap_class(cls: type) -> None:
  for name in list(vars(cls)):
    if name not in _AUTO_AE_METHODS:
      continue
    attr = vars(cls)[name]
    if callable(attr):
      setattr(cls, name, _ae(attr))


_wrap_class(MLXBackend)
