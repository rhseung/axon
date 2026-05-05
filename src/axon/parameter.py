from __future__ import annotations

from typing import Any, cast

from axon.backend import xp
from axon.backend.protocol import Array
from axon.dtype import DType
from axon.node import Node


class Parameter[D: DType](Node[D]):
  """학습 가능 leaf — `Node` 위에 영구 grad 버퍼를 추가한다.

  optimizer 가 step 사이에 누적/업데이트하는 `.grad` 를 소유. `Net.parameters()` 가
  `isinstance(value, Parameter)` 로 학습 대상만 모은다.
  """

  grad: Array[D]

  def __init__(self, data: Any, *, dtype: type[DType] = DType.FLOAT32):
    super().__init__(data, dtype=dtype)
    self._requires_grad = True
    self.zero_grad()

  def zero_grad(self):
    self.grad = cast(Array[D], xp.zeros_like(cast(Array, self._data)))
