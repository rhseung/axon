"""axon DType ↔ 백엔드 native dtype 매핑.

`to_backend_dtype(DType.FLOAT32)` → 현재 백엔드의 native float32 타입.
`from_backend_dtype(arr.dtype)` → axon DType subclass.

`DType` 은 class 계층이라 dtype 은 `type[DType]` (subclass) 으로 다룬다.
미지원 dtype 은 폴백 없이 즉시 TypeError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axon.dtype import DType

if TYPE_CHECKING:
  from axon.backend import BackendName


_UNSUPPORTED: dict[str, set[type[DType]]] = {
  "numpy": set(),
  # MLX 0.31 / Apple Metal: FLOAT64 는 GPU 에서 미지원 (CPU stream 에선 동작하지만
  # default device 가 GPU 라 사실상 막혀있다). INT64 는 모든 device 에서 지원.
  "mlx": {DType.FLOAT64},
  "cupy": {DType.BFLOAT16},
}


def _build_numpy_map() -> dict[type[DType], Any]:
  import numpy as np

  return {
    DType.FLOAT16: np.float16,
    DType.BFLOAT16: np.float32,  # NumPy 에 native bfloat16 없음 — fp32 로 대체
    DType.FLOAT32: np.float32,
    DType.FLOAT64: np.float64,
    DType.INT32: np.int32,
    DType.INT64: np.int64,
    DType.BOOL: np.bool_,
  }


def _build_mlx_map() -> dict[type[DType], Any]:
  import mlx.core as mx

  return {
    DType.FLOAT16: mx.float16,
    DType.BFLOAT16: mx.bfloat16,
    DType.FLOAT32: mx.float32,
    DType.INT32: mx.int32,
    DType.INT64: mx.int64,
    DType.BOOL: mx.bool_,
    # FLOAT64 는 _UNSUPPORTED 에서 차단 (GPU 미지원)
  }


def _build_cupy_map() -> dict[type[DType], Any]:
  import cupy as cp

  return {
    DType.FLOAT16: cp.float16,
    DType.FLOAT32: cp.float32,
    DType.FLOAT64: cp.float64,
    DType.INT32: cp.int32,
    DType.INT64: cp.int64,
    DType.BOOL: cp.bool_,
  }


_BUILDERS = {
  "numpy": _build_numpy_map,
  "mlx": _build_mlx_map,
  "cupy": _build_cupy_map,
}

_FORWARD_CACHE: dict[str, dict[type[DType], Any]] = {}
_REVERSE_CACHE: dict[str, dict[Any, type[DType]]] = {}


def _forward_map(name: str) -> dict[type[DType], Any]:
  if name not in _FORWARD_CACHE:
    _FORWARD_CACHE[name] = _BUILDERS[name]()
  return _FORWARD_CACHE[name]


def _reverse_map(name: str) -> dict[Any, type[DType]]:
  if name not in _REVERSE_CACHE:
    fwd = _forward_map(name)
    _REVERSE_CACHE[name] = {v: k for k, v in fwd.items()}
  return _REVERSE_CACHE[name]


def _resolve(backend_name: BackendName | None) -> BackendName:
  if backend_name is not None:
    return backend_name
  from axon.backend import get_backend

  return get_backend()


def to_backend_dtype(
  dtype: type[DType], backend_name: BackendName | None = None
) -> Any:
  """axon DType → 현재 백엔드의 실제 dtype 으로 변환.

  미지원 dtype 은 폴백 없이 즉시 TypeError.
  `backend_name` 을 명시하면 그 백엔드 기준으로 변환 (테스트 등).
  """
  name = _resolve(backend_name)
  if dtype in _UNSUPPORTED[name]:
    raise TypeError(
      f"{dtype.__name__} is not supported on the {name!r} backend. "
      f"Switch to a supported backend first: axon.set_backend('numpy')"
    )
  return _forward_map(name)[dtype]


def from_backend_dtype(
  native: Any, backend_name: BackendName | None = None
) -> type[DType]:
  """백엔드 native dtype → axon DType subclass. Tensor.dtype 구현에서 사용."""
  name = _resolve(backend_name)
  rev = _reverse_map(name)
  if native in rev:
    return rev[native]

  # numpy 의 경우 np.dtype 인스턴스로 들어올 수 있어 .type 으로 한 번 더 시도
  if name == "numpy" and hasattr(native, "type") and native.type in rev:
    return rev[native.type]

  raise TypeError(f"Unknown native dtype: {native!r} on backend {name!r}")
