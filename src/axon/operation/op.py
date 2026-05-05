from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from axon.dtype import DType
from axon.node import Node

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Op[D: DType](ABC):
  """연산 그래프 노드의 미분 규칙. forward/backward 는 순수 Array 산수, apply 가
  Node wrapping 을 담당한다.

  서브클래스가 구현하는 `forward` / `backward` 는 backend Array 만 다룬다 — 그래프
  metadata 는 무관. `apply` 가 입력 Node 들을 unwrap → forward 호출 → 결과 Array
  를 Node 로 wrap 하고, 입력 중 하나라도 `_requires_grad` 면 그래프 metadata 를
  세팅한다.

  주의: `forward` / `backward` 내부에서 Node 의 연산자 (`-x`, `x + y`, ...) 는
  쓰면 안 된다. Node 연산자는 다시 `Op.apply` 로 돌아오므로 무한 재귀.
  Array 의 연산자 (`a + b`) 는 backend native 산수라 OK.
  """

  @abstractmethod
  def forward(self, *inputs: Array[D]) -> Array[D]:
    """순전파 y = f(x_1, ..., x_n)."""
    ...

  @abstractmethod
  def backward(self, grad: Array[D], *inputs: Array[D]) -> tuple[Array[D], ...]:
    """체인룰로 입력별 손실 편미분 (∂L/∂x_1, ..., ∂L/∂x_n)."""
    ...

  def apply(self, *inputs: Node[D]) -> Node[D]:
    out_array = self.forward(*(n._data for n in inputs))
    out = Node.from_array(out_array)
    if any(n._requires_grad for n in inputs):
      out._op = self
      out._inputs = cast(tuple[Node, ...], inputs)
      out._requires_grad = True
    return out


class UnaryOp[D: DType](Op[D]):
  """단항 연산 기본 클래스."""

  @abstractmethod
  def forward_unary(self, x: Array[D]) -> Array[D]:
    """순전파 y = f(x)."""
    ...

  @abstractmethod
  def backward_unary(self, grad: Array[D], x: Array[D]) -> Array[D]:
    """역전파 ∂L/∂x."""
    ...

  def forward(self, *inputs: Array[D]) -> Array[D]:
    (x,) = inputs
    return self.forward_unary(x)

  def backward(self, grad: Array[D], *inputs: Array[D]) -> tuple[Array[D], ...]:
    (x,) = inputs
    return (self.backward_unary(grad, x),)


class BinaryOp[D: DType](Op[D]):
  """이항 연산: `forward_binary(a, b)`, `backward_binary(grad, a, b)` — `forward`/`backward`는 `*inputs`로 위임."""

  @abstractmethod
  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """순전파 y = f(a, b)."""
    ...

  @abstractmethod
  def backward_binary(
    self, grad: Array[D], a: Array[D], b: Array[D]
  ) -> tuple[Array[D], Array[D]]:
    """역전파 (∂L/∂a, ∂L/∂b)."""
    ...

  def forward(self, *inputs: Array[D]) -> Array[D]:
    a, b = inputs
    return self.forward_binary(a, b)

  def backward(
    self, grad: Array[D], *inputs: Array[D]
  ) -> tuple[Array[D], Array[D]]:
    a, b = inputs
    return self.backward_binary(grad, a, b)
