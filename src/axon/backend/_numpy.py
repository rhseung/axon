from __future__ import annotations

from typing import Any, cast

import numpy as np

from axon.backend._dtype import to_backend_dtype
from axon.backend.protocol import RandomProtocol
from axon.dtype import DType


class _NumpyRandom:
  def __init__(self) -> None:
    self._rng = np.random.default_rng()

  def normal(
    self,
    shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    out = self._rng.normal(loc=mean, scale=std, size=shape)
    return out.astype(to_backend_dtype(dtype, "numpy"))

  def uniform(
    self,
    shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    out = self._rng.uniform(low=low, high=high, size=shape)
    return out.astype(to_backend_dtype(dtype, "numpy"))

  def bernoulli(
    self,
    p: float,
    shape: tuple[int, ...],
    dtype: type[DType] = DType.FLOAT32,
  ):
    out = (self._rng.random(size=shape) < p).astype(to_backend_dtype(dtype, "numpy"))
    return out

  def seed(self, s: int) -> None:
    self._rng = np.random.default_rng(s)


class NumpyBackend:
  name: str = "numpy"
  random: RandomProtocol

  def __init__(self) -> None:
    # native 반환 타입 (np.ndarray) 이 Array Protocol 의 Any 반환과 구조적
    # 매칭이 안 되는 경우가 있어 cast 로 명시 위임 (BackendProtocol cast 와 동일 사유).
    self.random = cast(RandomProtocol, _NumpyRandom())

  # --- 생성 ---
  def array(self, data: Any, dtype: type[DType] = DType.FLOAT32):
    return np.asarray(data, dtype=to_backend_dtype(dtype, "numpy"))

  def zeros(self, shape: tuple[int, ...], dtype: type[DType] = DType.FLOAT32):
    return np.zeros(shape, dtype=to_backend_dtype(dtype, "numpy"))

  def ones(self, shape: tuple[int, ...], dtype: type[DType] = DType.FLOAT32):
    return np.ones(shape, dtype=to_backend_dtype(dtype, "numpy"))

  def full(
    self,
    shape: tuple[int, ...],
    fill_value: float,
    dtype: type[DType] = DType.FLOAT32,
  ):
    return np.full(shape, fill_value, dtype=to_backend_dtype(dtype, "numpy"))

  def zeros_like(self, x):
    return np.zeros_like(x)

  def ones_like(self, x):
    return np.ones_like(x)

  def full_like(self, x, fill_value: float):
    return np.full_like(x, fill_value)

  def eye(self, n: int, dtype: type[DType] = DType.FLOAT32):
    return np.eye(n, dtype=to_backend_dtype(dtype, "numpy"))

  def arange(
    self,
    start: float,
    stop: float | None = None,
    step: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    if stop is None:
      return np.arange(start, dtype=to_backend_dtype(dtype, "numpy"))
    return np.arange(start, stop, step, dtype=to_backend_dtype(dtype, "numpy"))

  # --- 수학 함수 ---
  def exp(self, x):
    return np.exp(x)

  def log(self, x):
    return np.log(x)

  def sqrt(self, x):
    return np.sqrt(x)

  def rsqrt(self, x):
    return 1.0 / np.sqrt(x)

  def abs(self, x):
    return np.abs(x)

  def sign(self, x):
    return np.sign(x)

  def maximum(self, a, b):
    return np.maximum(a, b)

  def minimum(self, a, b):
    return np.minimum(a, b)

  def clip(self, x, a_min, a_max):
    return np.clip(x, a_min, a_max)

  def sin(self, x):
    return np.sin(x)

  def cos(self, x):
    return np.cos(x)

  def tanh(self, x):
    return np.tanh(x)

  def power(self, a, b):
    return np.power(a, b)

  # --- 축소 ---
  def sum(self, x, axis=None, keepdims=False):
    return np.sum(x, axis=axis, keepdims=keepdims)

  def mean(self, x, axis=None, keepdims=False):
    return np.mean(x, axis=axis, keepdims=keepdims)

  def var(self, x, axis=None, keepdims=False, ddof=0):
    return np.var(x, axis=axis, keepdims=keepdims, ddof=ddof)

  def max(self, x, axis=None, keepdims=False):
    return np.max(x, axis=axis, keepdims=keepdims)

  def min(self, x, axis=None, keepdims=False):
    return np.min(x, axis=axis, keepdims=keepdims)

  def norm(self, x, ord=None, axis=None, keepdims=False):
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

  # --- 형상 ---
  def reshape(self, x, shape):
    return np.reshape(x, shape)

  def transpose(self, x, axes=None):
    return np.transpose(x, axes)

  def expand_dims(self, x, axis):
    return np.expand_dims(x, axis)

  def squeeze(self, x, axis=None):
    return np.squeeze(x, axis)

  def broadcast_to(self, x, shape):
    return np.broadcast_to(x, shape)

  def concatenate(self, arrays, axis=0):
    return np.concatenate(arrays, axis=axis)

  def stack(self, arrays, axis=0):
    return np.stack(arrays, axis=axis)

  def split(self, x, indices_or_sections, axis=0):
    return list(np.split(x, indices_or_sections, axis=axis))

  def flip(self, x, axis=None):
    return np.flip(x, axis=axis)

  # --- 선형대수 ---
  def matmul(self, a, b):
    return np.matmul(a, b)

  def einsum(self, subscripts, *operands):
    return np.einsum(subscripts, *operands)

  # --- 인덱싱 ---
  def where(self, condition, x, y):
    return np.where(condition, x, y)

  def take(self, x, indices, axis=None):
    return np.take(x, indices, axis=axis)

  def tril(self, x, k=0):
    return np.tril(x, k=k)

  def triu(self, x, k=0):
    return np.triu(x, k=k)

  # --- 정렬 / argmax 류 ---
  def sort(self, x, axis=-1):
    return np.sort(x, axis=axis)

  def argsort(self, x, axis=-1):
    return np.argsort(x, axis=axis)

  def argmax(self, x, axis=None, keepdims=False):
    return np.argmax(x, axis=axis, keepdims=keepdims)

  def argmin(self, x, axis=None, keepdims=False):
    return np.argmin(x, axis=axis, keepdims=keepdims)

  # --- 누적 ---
  def cumsum(self, x, axis=None):
    return np.cumsum(x, axis=axis)

  # --- 패딩 ---
  def pad(self, x, pad_width, constant_values=0.0):
    return np.pad(x, pad_width, mode="constant", constant_values=constant_values)

  # --- 변환 / 평가 ---
  def to_numpy(self, x):
    return np.asarray(x)

  def from_numpy(self, x):
    return np.asarray(x)

  def eval(self, *arrays):
    pass  # NumPy 는 eager — no-op

  def async_eval(self, *arrays):
    pass  # NumPy 는 eager — no-op
