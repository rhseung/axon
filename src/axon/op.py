from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import DTypeLike

from axon.tensor import Tensor


class Op[D: DTypeLike](ABC):
  def __call__(self, *inputs: Tensor[D]):
    """연산을 적용해 y를 만들고, 역전파용 그래프 정보를 연결한다."""
    y = self.forward(*inputs)
    y._op = self
    y._inputs = inputs
    return y

  @abstractmethod
  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    """순전파 y = f(x_1, ..., x_n)를 계산한다."""
    ...

  @abstractmethod
  def backward(self, grad: Tensor[D], *inputs: Tensor[D]) -> tuple[Tensor[D], ...]:
    """체인룰로 입력별 손실 편미분을 계산한다.

    기호:
      - L: 최종 스칼라 손실
      - y = f(x_1, x_2, ..., x_n): 현재 Op의 출력

    Args:
      grad:
        상류(upstream)에서 전달된 편미분으로, 수식으로는
        grad = ∂L/∂y 를 의미한다.
      *inputs:
        순전파 `forward`에 실제로 전달되었던 입력 텐서들
        (x_1, x_2, ..., x_n) 이다.

    Returns:
      입력 순서를 유지한 튜플
      (∂L/∂x_1, ∂L/∂x_2, ..., ∂L/∂x_n).
      각 원소는 입력 텐서와 같은 shape의 `Tensor`다.
    """
    ...


class Identity[D: DTypeLike](Op[D]):
  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    """항등 연산 y = x 를 계산한다."""
    (x,) = inputs
    return x.copy()

  def backward(self, grad: Tensor[D], *inputs: Tensor[D]):
    """항등 연산의 체인룰을 계산한다: ∂L/∂x = (∂L/∂y)(∂y/∂x), ∂y/∂x = 1."""
    (x,) = inputs

    dy_dx = Tensor.ones_like(x)

    return (Tensor(grad._data * dy_dx._data),)


class Add[D: DTypeLike](Op[D]):
  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    """덧셈 순전파 y = x1 + x2 를 계산한다."""
    x1, x2 = inputs
    return Tensor(x1._data + x2._data)

  def backward(self, grad: Tensor[D], *inputs: Tensor[D]):
    """덧셈의 체인룰을 계산한다.

    y = x1 + x2 이므로 ∂y/∂x1 = 1, ∂y/∂x2 = 1 이고,
    따라서 ∂L/∂x1 = (∂L/∂y)(∂y/∂x1), ∂L/∂x2 = (∂L/∂y)(∂y/∂x2).
    """
    x1, x2 = inputs

    dy_dx1 = Tensor.ones_like(x1)
    dy_dx2 = Tensor.ones_like(x2)

    return (
      Tensor(grad._data * dy_dx1._data),
      Tensor(grad._data * dy_dx2._data),
    )


class Mul[D: DTypeLike](Op[D]):
  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    """곱셈 순전파 y = x1 * x2 를 계산한다."""
    x1, x2 = inputs
    return Tensor(x1._data * x2._data)

  def backward(self, grad: Tensor[D], *inputs: Tensor[D]):
    """곱셈의 체인룰을 계산한다.

    y = x1 * x2 이므로 ∂y/∂x1 = x2, ∂y/∂x2 = x1 이고,
    따라서 ∂L/∂x1 = (∂L/∂y)(∂y/∂x1), ∂L/∂x2 = (∂L/∂y)(∂y/∂x2).
    """
    x1, x2 = inputs

    dy_dx1 = x2
    dy_dx2 = x1

    return (
      Tensor(grad._data * dy_dx1._data),
      Tensor(grad._data * dy_dx2._data),
    )


class MatMul[D: DTypeLike](Op[D]):
  def forward(self, *inputs: Tensor[D]) -> Tensor[D]:
    """행렬 곱셈 순전파 y = x1 @ x2 를 계산한다."""
    x1, x2 = inputs
    return Tensor(x1._data @ x2._data)

  def backward(self, grad: Tensor[D], *inputs: Tensor[D]):
    """행렬 곱셈의 체인룰을 계산한다.

    y = x1 @ x2 이므로 ∂L/∂x1 = (∂L/∂y) @ x2.T, ∂L/∂x2 = x1.T @ (∂L/∂y).
    """
    x1, x2 = inputs

    return (
      Tensor(grad._data @ x2._data.T),
      Tensor(x1._data.T @ grad._data),
    )
