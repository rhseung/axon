from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from axon.operation.binary.base import BinaryOp
from axon.tensor import Tensor


class Pow[D: DTypeLike](BinaryOp[D]):
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """거듭제곱 순전파 y = a ** b (원소별) 를 계산한다."""
    return Tensor(a._data**b._data)

  def backward_binary(self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]) -> tuple[Tensor[D], Tensor[D]]:
    """y = a ** b 의 체인룰 (원소별).

    ∂y/∂a = b * a ** (b - 1), ∂y/∂b = a ** b * log(a).
    """
    y_data = a._data**b._data
    dy_da = b._data * np.power(a._data, b._data - 1)
    dy_db = y_data * np.log(a._data)
    return (
      Tensor(grad._data * dy_da),
      Tensor(grad._data * dy_db),
    )
