from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import Op

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Add[D: DType](Op[D]):
  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = a + b."""
    a, b = inputs
    return a + b

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂y/∂a = ∂y/∂b = 1."""
    return (grad, grad)
