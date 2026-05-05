from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import BinaryOp

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Div[D: DType](BinaryOp[D]):
  """`mul(a, pow(b, -1))` 로 합성하지 않고 primitive 로 둔 이유:
  Pow 의 backward 가 `xp.log(a)` 를 쓰므로, 그 경로로 가면 b 에 음수가 하나라도
  있을 때 그래디언트가 NaN 으로 오염된다. `1/b` 자체는 음수에서 멀쩡한 함수이므로
  Div 를 직접 정의해야 수치적으로 안전하다 (성능도 reciprocal 이 더 빠름).
  """

  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """나눗셈 순전파 y = a / b (원소별) 를 계산한다."""
    return a / b

  def backward_binary(
    self, grad: Array[D], a: Array[D], b: Array[D]
  ) -> tuple[Array[D], Array[D]]:
    """y = a / b 의 체인룰 (원소별).

    ∂y/∂a = 1 / b, ∂y/∂b = -a / b**2.
    """
    return (grad / b, -grad * a / (b * b))
