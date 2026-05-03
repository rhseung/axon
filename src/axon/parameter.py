from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from axon.tensor import Tensor


class Parameter[D: DTypeLike](Tensor[D]):
  grad: Tensor[D]

  def __init__(self, data: NDArray[Any], *, dtype: DTypeLike = np.float32):
    super().__init__(data, dtype=dtype)
    self.zero_grad()

  def zero_grad(self):
    self.grad = Tensor.zeros_like(self)
