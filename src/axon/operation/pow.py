from __future__ import annotations

from typing import cast

from axon.backend import xp
from axon.backend.protocol import Array
from axon.dtype import DType
from axon.operation.op import Op


class Pow[D: DType](Op[D]):
  """y = a ** b. b 가 `Constant` (Scalar 지수 등) 면 `Var.backward` 가 db 를
  폐기 — log(a) 가 음수에서 NaN 이라도 그래프에 영향 없음 (계산만 낭비).
  """

  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = a ** b (원소별)."""
    a, b = inputs
    return a**b

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂y/∂a = b·a^(b-1), ∂y/∂b = a^b · log(a)."""
    a, b = inputs
    da = grad * b * a ** (b - 1)
    db = grad * a**b * cast(Array[D], xp.log(cast(Array, a)))
    return (da, db)
