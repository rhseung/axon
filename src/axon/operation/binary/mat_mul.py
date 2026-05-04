from __future__ import annotations

from numpy.typing import DTypeLike

from axon.operation.binary.base import BinaryOp
from axon.tensor import Tensor


class MatMul[D: DTypeLike](BinaryOp[D]):
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """행렬 곱셈 순전파 y = a @ b 를 계산한다."""
    return Tensor(a._data @ b._data)

  def backward_binary(self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]) -> tuple[Tensor[D], Tensor[D]]:
    """행렬 곱셈의 체인룰. ∂L/∂a = (∂L/∂y) @ b.T, ∂L/∂b = a.T @ (∂L/∂y)."""
    return (
      Tensor(grad._data @ b._data.T),
      Tensor(a._data.T @ grad._data),
    )
