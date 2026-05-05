from __future__ import annotations

from typing import Any

from axon.dtype import DType
from axon.tensor import Tensor


class Parameter[D: DType](Tensor[D]):
  grad: Tensor[D]

  def __init__(self, data: Any, *, dtype: type[DType] = DType.FLOAT32):
    super().__init__(data, dtype=dtype)
    self.zero_grad()

  def zero_grad(self):
    self.grad = Tensor.zeros_like(self)
