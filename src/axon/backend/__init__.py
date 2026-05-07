"""백엔드 진입점. `xp` 가 라이브 프록시 — `set_backend()` 후 같은 변수 재사용 OK."""

from __future__ import annotations

from typing import Literal, cast

from axon.backend.protocol import Array, BackendProtocol, RandomProtocol

BackendName = Literal["numpy", "mlx", "cupy"]


_current: BackendProtocol | None = None
_name: BackendName = "numpy"


def set_backend(name: BackendName = "numpy") -> None:
  global _current, _name

  match name:
    case "numpy":
      from axon.backend._numpy import NumpyBackend

      _current = cast(BackendProtocol, NumpyBackend())
    case "mlx":
      from axon.backend._mlx import MLXBackend

      _current = cast(BackendProtocol, MLXBackend())
    case "cupy":
      from axon.backend._cupy import CuPyBackend

      _current = cast(BackendProtocol, CuPyBackend())
    case _:
      raise ValueError(f"알 수 없는 백엔드: {name!r}")

  _name = name


def get_backend() -> BackendName:
  return _name


def current() -> BackendProtocol:
  if _current is None:
    set_backend("numpy")
  assert _current is not None
  return _current


class _BackendProxy:
  """현재 백엔드에 모든 속성 접근 위임 — 라이브 변수."""

  __slots__ = ()

  def __getattribute__(self, name: str):
    return getattr(current(), name)


xp: BackendProtocol = cast(BackendProtocol, _BackendProxy())


__all__ = [
  "Array",
  "BackendName",
  "BackendProtocol",
  "RandomProtocol",
  "current",
  "get_backend",
  "set_backend",
  "xp",
]
