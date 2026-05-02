from __future__ import annotations
from numpy.typing import NDArray


from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Any

from axon.tensor import Tensor


class Op(ABC):
  @abstractmethod
  def forward(self, *inputs: Tensor[Any]) -> Tensor[Any]: ...

  @abstractmethod
  def backward(self, grad: Any, *inputs: tuple[Tensor, ...]) -> None: ...

  def _make_output(
    self, *, inputs: tuple[Tensor[Any], ...], output: NDArray[Any]
  ) -> Tensor[Any]:
    """``fn(*inputs)`` 로 ndarray를 만들고, 같은 ``inputs`` 를 ``_inputs`` 에 넣는다."""
    out = deepcopy(output)
    out._op = self
    out._inputs = inputs
    return out


class Identity(Op):
  def forward(self, *inputs: Tensor[Any]) -> Tensor[Any]:
    (x,) = inputs
    return self._make_output(inputs=inputs, output=x)

  def backward(self, grad: Any, *inputs: tuple[Tensor, ...]) -> None:
    pass


class Add(Op):
  def forward(self, *inputs: Tensor[Any]) -> Tensor[Any]:
    x, y = inputs
    return self._make_output(inputs=inputs, output=x)

  def backward(self, grad: Any, *inputs: tuple[Tensor, ...]) -> None:
    pass

  def forward(self, a, b):
    out = Tensor(a.data + b.data)
    return self._make_output(inputs=(a, b), output=out)

  def backward(self, grad, inputs):
    a, b = inputs[0]  # ← 시그니처 확인 필요
    a.grad += grad.data  # ∂(a+b)/∂a = 1
    b.grad += grad.data  # ∂(a+b)/∂b = 1
