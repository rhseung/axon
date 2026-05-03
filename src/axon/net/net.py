from abc import ABC, abstractmethod
from typing import Any

from axon.parameter import Parameter
from axon.tensor import Tensor


class Net(ABC):
  @abstractmethod
  def forward(self, x: Tensor[Any]) -> Tensor[Any]: ...

  def parameters(self) -> list[Parameter]:
    parameters: list[Parameter] = []
    visited: set[int] = set()

    def collect(value: Any):
      value_id = id(value)
      if value_id in visited:
        return
      visited.add(value_id)

      if isinstance(value, Parameter):
        parameters.append(value)
        return

      if isinstance(value, Net):
        for child in value.__dict__.values():
          collect(child)
        return

      if isinstance(value, dict):
        for child in value.values():
          collect(child)
        return

      if isinstance(value, (list, tuple)):
        for child in value:
          collect(child)

    collect(self)
    return parameters
