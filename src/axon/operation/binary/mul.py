from __future__ import annotations

from numpy.typing import DTypeLike

from axon.operation.binary.base import BinaryOp
from axon.tensor import Tensor


class Mul[D: DTypeLike](BinaryOp[D]):
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """곱셈 순전파 y = a * b 를 계산한다."""
    return Tensor(a._data * b._data)

  def backward_binary(
    self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]
  ) -> tuple[Tensor[D], Tensor[D]]:
    """곱셈의 체인룰. y = a * b 이므로 ∂y/∂a = b, ∂y/∂b = a."""
    return (
      Tensor(grad._data * b._data),
      Tensor(grad._data * a._data),
    )
