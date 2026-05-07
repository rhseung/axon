from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import Op

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Neg[D: DType](Op[D]):
  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = -x."""
    (x,) = inputs
    return -x

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂y/∂x = -1."""
    return (-grad,)
