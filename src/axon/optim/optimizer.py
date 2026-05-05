from abc import ABC, abstractmethod
from copy import deepcopy

from axon.parameter import Parameter


class Optimizer(ABC):
  def __init__(self, parameters: list[Parameter], lr: float):
    self._parameters = parameters
    self._lr = lr

  @abstractmethod
  def optimize(self, p: Parameter): ...

  @property
  def lr(self):
    return self._lr

  @property
  def parameters(self):
    return deepcopy(self._parameters)

  def zero_grad(self):
    for p in self._parameters:
      p.zero_grad()

  def step(self):
    for p in self._parameters:
      self.optimize(p)
