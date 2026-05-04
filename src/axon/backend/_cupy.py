"""CuPy 백엔드.

NumPy 와 API 가 거의 동일하므로 위임 패턴이 단순하다.
로컬 (macOS) 에서 검증 불가 — Colab / Windows / 외장 GPU 환경에서 `pytest -m cuda`.
"""

from __future__ import annotations

from typing import Any, cast

from axon.backend._dtype import to_backend_dtype
from axon.backend.protocol import RandomProtocol
from axon.dtype import DType


class _CuPyRandom:
  def __init__(self) -> None:
    import cupy as cp

    self._rng = cp.random.default_rng()

  def normal(
    self,
    shape: tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    out = self._rng.normal(loc=mean, scale=std, size=shape)
    return out.astype(to_backend_dtype(dtype, "cupy"))

  def uniform(
    self,
    shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
    dtype: type[DType] = DType.FLOAT32,
  ):
    out = self._rng.uniform(low=low, high=high, size=shape)
    return out.astype(to_backend_dtype(dtype, "cupy"))

  def bernoulli(
    self,
    p: float,
    shape: tuple[int, ...],
    dtype: type[DType] = DType.FLOAT32,
  ):
    out = (self._rng.random(size=shape) < p).astype(to_backend_dtype(dtype, "cupy"))
    return out

  def seed(self, s: int) -> None:
    import cupy as cp

    self._rng = cp.random.default_rng(s)


class CuPyBackend:
  name: str = "cupy"
  random: RandomProtocol

  def __init__(self) -> None:
    import cupy as cp

    self._cp = cp
    self.random = cast(RandomProtocol, _CuPyRandom())

  # --- 생성 ---
  def array(self, data: Any, dtype: type[DType] = DType.FLOAT32):
    return self._cp.asarray(data, dtype=to_backend_dtype(dtype, "cupy"))

  def zeros(self, shape, dtype: type[DType] = DType.FLOAT32):
    return self._cp.zeros(shape, dtype=to_backend_dtype(dtype, "cupy"))

  def ones(self, shape, dtype: type[DType] = DType.FLOAT32):
    return self._cp.ones(shape, dtype=to_backend_dtype(dtype, "cupy"))

  def full(self, shape, fill_value, dtype: type[DType] = DType.FLOAT32):
    return self._cp.full(shape, fill_value, dtype=to_backend_dtype(dtype, "cupy"))

  def zeros_like(self, x):
    return self._cp.zeros_like(x)

  def ones_like(self, x):
    return self._cp.ones_like(x)

  def full_like(self, x, fill_value):
    return self._cp.full_like(x, fill_value)

  def eye(self, n, dtype: type[DType] = DType.FLOAT32):
    return self._cp.eye(n, dtype=to_backend_dtype(dtype, "cupy"))

  def arange(self, start, stop=None, step=1.0, dtype: type[DType] = DType.FLOAT32):
    if stop is None:
      return self._cp.arange(start, dtype=to_backend_dtype(dtype, "cupy"))
    return self._cp.arange(start, stop, step, dtype=to_backend_dtype(dtype, "cupy"))

  # --- 수학 함수 ---
  def exp(self, x):
    return self._cp.exp(x)

  def log(self, x):
    return self._cp.log(x)

  def sqrt(self, x):
    return self._cp.sqrt(x)

  def rsqrt(self, x):
    return 1.0 / self._cp.sqrt(x)

  def abs(self, x):
    return self._cp.abs(x)

  def sign(self, x):
    return self._cp.sign(x)

  def maximum(self, a, b):
    return self._cp.maximum(a, b)

  def minimum(self, a, b):
    return self._cp.minimum(a, b)

  def clip(self, x, a_min, a_max):
    return self._cp.clip(x, a_min, a_max)

  def sin(self, x):
    return self._cp.sin(x)

  def cos(self, x):
    return self._cp.cos(x)

  def tanh(self, x):
    return self._cp.tanh(x)

  def power(self, a, b):
    return self._cp.power(a, b)

  # --- 축소 ---
  def sum(self, x, axis=None, keepdims=False):
    return self._cp.sum(x, axis=axis, keepdims=keepdims)

  def mean(self, x, axis=None, keepdims=False):
    return self._cp.mean(x, axis=axis, keepdims=keepdims)

  def var(self, x, axis=None, keepdims=False, ddof=0):
    return self._cp.var(x, axis=axis, keepdims=keepdims, ddof=ddof)

  def max(self, x, axis=None, keepdims=False):
    return self._cp.max(x, axis=axis, keepdims=keepdims)

  def min(self, x, axis=None, keepdims=False):
    return self._cp.min(x, axis=axis, keepdims=keepdims)

  def norm(self, x, ord=None, axis=None, keepdims=False):
    return self._cp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

  # --- 형상 ---
  def reshape(self, x, shape):
    return self._cp.reshape(x, shape)

  def transpose(self, x, axes=None):
    return self._cp.transpose(x, axes)

  def expand_dims(self, x, axis):
    return self._cp.expand_dims(x, axis)

  def squeeze(self, x, axis=None):
    return self._cp.squeeze(x, axis)

  def broadcast_to(self, x, shape):
    return self._cp.broadcast_to(x, shape)

  def concatenate(self, arrays, axis=0):
    return self._cp.concatenate(arrays, axis=axis)

  def stack(self, arrays, axis=0):
    return self._cp.stack(arrays, axis=axis)

  def split(self, x, indices_or_sections, axis=0):
    return list(self._cp.split(x, indices_or_sections, axis=axis))

  def flip(self, x, axis=None):
    return self._cp.flip(x, axis=axis)

  # --- 선형대수 ---
  def matmul(self, a, b):
    return self._cp.matmul(a, b)

  def einsum(self, subscripts, *operands):
    return self._cp.einsum(subscripts, *operands)

  # --- 인덱싱 ---
  def where(self, condition, x, y):
    return self._cp.where(condition, x, y)

  def take(self, x, indices, axis=None):
    return self._cp.take(x, indices, axis=axis)

  def tril(self, x, k=0):
    return self._cp.tril(x, k=k)

  def triu(self, x, k=0):
    return self._cp.triu(x, k=k)

  # --- 정렬 / argmax 류 ---
  def sort(self, x, axis=-1):
    return self._cp.sort(x, axis=axis)

  def argsort(self, x, axis=-1):
    return self._cp.argsort(x, axis=axis)

  def argmax(self, x, axis=None, keepdims=False):
    return self._cp.argmax(x, axis=axis, keepdims=keepdims)

  def argmin(self, x, axis=None, keepdims=False):
    return self._cp.argmin(x, axis=axis, keepdims=keepdims)

  # --- 누적 ---
  def cumsum(self, x, axis=None):
    return self._cp.cumsum(x, axis=axis)

  # --- 패딩 ---
  def pad(self, x, pad_width, constant_values=0.0):
    return self._cp.pad(x, pad_width, mode="constant", constant_values=constant_values)

  # --- 변환 / 평가 ---
  def to_numpy(self, x):
    return self._cp.asnumpy(x)

  def from_numpy(self, x):
    return self._cp.asarray(x)

  def eval(self, *arrays):
    pass  # CuPy 도 eager

  def async_eval(self, *arrays):
    pass  # CuPy 도 eager — no-op
