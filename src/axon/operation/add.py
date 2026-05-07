from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import BinaryOp

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Add[D: DType](BinaryOp[D]):
  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """덧셈 순전파 y = a + b 를 계산한다."""
    return a + b

  def backward_binary(
    self,
    grad: Array[D],
    a: Array[D],
    b: Array[D],
    *,
    needs_grad: tuple[bool, bool],
  ) -> tuple[Array[D] | None, Array[D] | None]:
    """덧셈의 체인룰. y = a + b 이므로 ∂y/∂a = ∂y/∂b = 1, grad 그대로 전달."""
    return (
      grad if needs_grad[0] else None,
      grad if needs_grad[1] else None,
    )
