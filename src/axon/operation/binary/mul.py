from __future__ import annotations

from axon.dtype import DType
from axon.operation.binary.base import BinaryOp
from axon.tensor import Tensor


class Mul[D: DType](BinaryOp[D]):
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """곱셈 순전파 y = a * b 를 계산한다."""
    return Tensor.from_array(a._data * b._data)

  def backward_binary(
    self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]
  ) -> tuple[Tensor[D], Tensor[D]]:
    """곱셈의 체인룰. y = a * b 이므로 ∂y/∂a = b, ∂y/∂b = a."""
    return (
      Tensor.from_array(grad._data * b._data),
      Tensor.from_array(grad._data * a._data),
    )
