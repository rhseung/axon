from __future__ import annotations

from abc import abstractmethod

from axon.dtype import DType
from axon.operation.op import Op
from axon.tensor import Tensor


class BinaryOp[D: DType](Op[D]):
  """이항 연산: `forward_binary(a, b)`, `backward_binary(grad, a, b)` — `forward`/`backward`는 `*inputs`로 위임."""

  @abstractmethod
  def forward_binary(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
    """순전파 y = f(a, b)."""
    ...

  @abstractmethod
  def backward_binary(
    self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]
  ) -> tuple[Tensor[D], Tensor[D]]:
    """역전파 (∂L/∂a, ∂L/∂b)."""
    ...

  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    a, b = inputs
    return self.forward_binary(a, b)

  def backward(
    self, grad: Tensor[D], *inputs: Tensor[D]
  ) -> tuple[Tensor[D], Tensor[D]]:
    a, b = inputs
    return self.backward_binary(grad, a, b)
