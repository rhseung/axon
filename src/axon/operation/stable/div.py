from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import Op

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Div[D: DType](Op[D]):
  """`mul(a, pow(b, -1))` 합성을 피한다 — Pow.backward 의 `xp.log(a)` 가 음수
  base 에서 NaN 을 만들기 때문. `1/b` 는 음수에서 멀쩡해 직접 primitive 가 안전.
  """

  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = a / b (원소별)."""
    a, b = inputs
    return a / b

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂y/∂a = 1/b, ∂y/∂b = -a/b²."""
    a, b = inputs
    return (grad / b, -grad * a / (b * b))
