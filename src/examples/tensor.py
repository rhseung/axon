from __future__ import annotations

from typing import Any, overload

import numpy as np
from numpy.typing import NDArray


def _merge_grad(
  existing: NDArray[Any] | int | float,
  delta: NDArray[Any],
) -> NDArray[Any]:
  if isinstance(existing, (int, float)) and existing == 0:
    return delta
  assert isinstance(existing, np.ndarray)
  return existing + delta


class Tensor[D: np.generic]:
  def __init__(self, data: NDArray[D]):
    self.data = data
    self.grad: NDArray[D] | int | float = 0  # 첫 backward에서 ndarray로 교체됨
    self._backward = lambda: None
    self._prev_inputs: list[Tensor[Any]] = []

  @overload
  def __add__(self, other: Tensor[Any]) -> Tensor[Any]: ...

  @overload
  def __add__(self, other: np.generic) -> Tensor[Any]: ...

  @overload
  def __add__(self, other: int | float) -> Tensor[Any]: ...

  def __add__(self, other: Any) -> Tensor[Any]:
    match other:
      case Tensor() as rhs:
        assert self.data.shape == rhs.data.shape, (
          f"덧셈은 같은 shape만 허용합니다: {self.data.shape} vs {rhs.data.shape}"
        )
        out = Tensor[Any](self.data + rhs.data)
        out._prev_inputs = [self, rhs]

        def _backward() -> None:
          go = np.asarray(out.grad)
          self.grad = _merge_grad(self.grad, go)
          rhs.grad = _merge_grad(rhs.grad, go)

        out._backward = _backward
        return out

      case _:
        arr = np.asarray(other)
        assert self.data.shape == arr.shape, (
          f"덧셈은 같은 shape만 허용합니다: {self.data.shape} vs {arr.shape}"
        )
        out = Tensor[Any](self.data + arr)
        out._prev_inputs = [self]

        def _backward() -> None:
          go = np.asarray(out.grad)
          self.grad = _merge_grad(self.grad, go)

        out._backward = _backward
        return out

  def __radd__(self, other: Any) -> Tensor[Any]:
    return self.__add__(other)
