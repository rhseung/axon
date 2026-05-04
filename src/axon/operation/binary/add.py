from __future__ import annotations

from numpy.typing import DTypeLike

from axon.operation.binary.base import BinaryOp
from axon.tensor import Tensor


class Add[D: DTypeLike](BinaryOp[D]):
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """덧셈 순전파 y = a + b 를 계산한다."""
    return Tensor(a._data + b._data)

  def backward_binary(
    self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]
  ) -> tuple[Tensor[D], Tensor[D]]:
    """덧셈의 체인룰. y = a + b 이므로 ∂y/∂a = ∂y/∂b = 1."""
    dy_da = Tensor.ones_like(a)
    dy_db = Tensor.ones_like(b)
    return (
      Tensor(grad._data * dy_da._data),
      Tensor(grad._data * dy_db._data),
    )
