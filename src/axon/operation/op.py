from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from axon.dtype import DType

if TYPE_CHECKING:
  from axon.tensor import Tensor


class Op[D: DType](ABC):
  """연산 그래프 노드. 순전파·역전파 모두 `*inputs` 규약으로 `Op`에 두고, 서브클래스는 `_unary` / `_binary` 본체를 구현한다."""

  @abstractmethod
  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    """순전파 y = f(x_1, ..., x_n)."""
    ...

  @abstractmethod
  def backward(self, grad: Tensor[D], *inputs: Tensor[D]) -> tuple[Tensor[D], ...]:
    """체인룰로 입력별 손실 편미분 (∂L/∂x_1, ..., ∂L/∂x_n)."""
    ...

  def __call__(self, *inputs: Tensor[D]) -> Tensor[D]:
    y = self.forward(*inputs)
    y._op = self
    y._inputs = cast(tuple[Tensor, ...], inputs)
    return y
