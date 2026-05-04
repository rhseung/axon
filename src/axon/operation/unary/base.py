from __future__ import annotations

from abc import abstractmethod

from numpy.typing import DTypeLike

from axon.operation.op import Op
from axon.tensor import Tensor


class UnaryOp[D: DTypeLike](Op[D]):
  """단항 연산 기본 클래스."""

  @abstractmethod
  def forward_unary(self, x: Tensor[D]) -> Tensor[D]:
    """순전파 y = f(x)."""
    ...

  @abstractmethod
  def backward_unary(self, grad: Tensor[D], x: Tensor[D]) -> Tensor[D]:
    """역전파 ∂L/∂x."""
    ...

  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    (x,) = inputs
    return self.forward_unary(x)

  def backward(self, grad: Tensor[D], *inputs: Tensor[D]) -> tuple[Tensor[D], ...]:
    (x,) = inputs
    return (self.backward_unary(grad, x),)
