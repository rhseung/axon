from __future__ import annotations

from typing import cast

from axon.backend import xp
from axon.backend.protocol import Array
from axon.dtype import DType
from axon.operation.op import Op


class Sigmoid[D: DType](Op[D]):
  """`y = 1 / (1 + exp(-x))`. closed-form `y(1-y)` backward 살리려고 fused."""

  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = 1 / (1 + exp(-x))."""
    (x,) = inputs
    return cast(Array[D], 1 / (1 + xp.exp(cast(Array, -x))))

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂y/∂x = y · (1 − y)."""
    (x,) = inputs
    y = cast(Array[D], 1 / (1 + xp.exp(cast(Array, -x))))
    return (grad * y * (1 - y),)
