from __future__ import annotations

from typing import TYPE_CHECKING

from axon.backend import xp
from axon.optim.optimizer import Optimizer

if TYPE_CHECKING:
  from axon.backend.protocol import Array
  from axon.var import Var


class SGD(Optimizer):
  """SGD with optional momentum."""

  def __init__(self, lr: float, momentum: float = 0.0):
    self.lr = lr
    self.momentum = momentum
    self._velocity: Array | None = None

  def update(self, var: Var) -> None:
    if self.momentum > 0:
      if self._velocity is None:
        self._velocity = xp.zeros_like(var._data)
      self._velocity = self.momentum * self._velocity + var.grad
      var._data = var._data - self.lr * self._velocity
    else:
      var._data = var._data - self.lr * var.grad
