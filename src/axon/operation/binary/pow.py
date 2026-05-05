from __future__ import annotations

from axon.backend import xp
from axon.dtype import DType
from axon.operation.binary.base import BinaryOp
from axon.tensor import Tensor


class Pow[D: DType](BinaryOp[D]):
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """거듭제곱 순전파 y = a ** b (원소별) 를 계산한다."""
    return Tensor.from_array(a._data**b._data)

  def backward_binary(
    self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]
  ) -> tuple[Tensor[D], Tensor[D]]:
    """y = a ** b 의 체인룰 (원소별).

    ∂y/∂a = b * a ** (b - 1), ∂y/∂b = a ** b * log(a).
    """
    y = a._data**b._data
    dy_da = b._data * a._data ** (b._data - 1)
    dy_db = y * xp.log(a._data)
    return (
      Tensor.from_array(grad._data * dy_da),
      Tensor.from_array(grad._data * dy_db),
    )
