"""백엔드 추상화 진입점.

`axon.set_backend("mlx")` 한 줄로 NumPy / MLX / CuPy 백엔드 전환.
`from axon.backend import xp` 후 `xp.exp(x)` 처럼 NumPy 와 동일한 패턴으로 사용.

xp 는 `_BackendProxy` 인스턴스 — `__getattribute__` 로 매번 `current()` 에 위임하므로
`set_backend()` 후에도 같은 변수가 새 백엔드를 가리킨다. 정적 타입은 `BackendProtocol`
로 보이므로 pyrefly / IDE 가 모든 멤버를 추론한다.
"""

from __future__ import annotations

from typing import Literal, cast

from axon.backend.protocol import Array, BackendProtocol, RandomProtocol

BackendName = Literal["numpy", "mlx", "cupy"]


_current: BackendProtocol | None = None
_name: BackendName = "numpy"


def set_backend(name: BackendName = "numpy") -> None:
  """백엔드 전환.

  concrete 백엔드 인스턴스를 `BackendProtocol` 로 cast 한다 — Protocol 의 반환 타입
  (`Array`, `list[Array]` 등) 과 native 반환 타입 (`np.ndarray`, `list[np.ndarray]`) 의
  구조적 차이를 감수하고 명시 위임. 사용자는 `xp.method` 시 BackendProtocol 시그니처로
  IDE 추론을 받는다.
  """
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
  """현재 백엔드 인스턴스에 모든 속성 접근을 위임한다.

  `xp.exp(x)` → `current().exp(x)` 자동 위임.
  `set_backend()` 이후에도 같은 `xp` 변수가 새 백엔드를 가리킨다.
  """

  __slots__ = ()

  def __getattribute__(self, name: str):
    return getattr(current(), name)


# 정적 타입은 BackendProtocol — pyrefly / IDE 가 멤버 추론. 런타임은 프록시.
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
