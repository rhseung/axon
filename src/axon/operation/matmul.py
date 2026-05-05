from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import BinaryOp

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class MatMul[D: DType](BinaryOp[D]):
  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """행렬 곱셈 순전파 y = a @ b 를 계산한다."""
    return a @ b

  def backward_binary(
    self, grad: Array[D], a: Array[D], b: Array[D]
  ) -> tuple[Array[D], Array[D]]:
    """행렬 곱셈의 체인룰. ∂L/∂a = (∂L/∂y) @ b.T, ∂L/∂b = a.T @ (∂L/∂y)."""
    return (grad @ b.T, a.T @ grad)
