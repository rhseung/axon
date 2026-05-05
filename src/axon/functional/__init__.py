"""연산 API.

현재 구현 완료된 primitive는 `add`, `mul`, `matmul`, `pow`만 지원한다.
나머지 API는 시그니처만 유지하고 `NotImplementedError`를 던진다.
"""

from __future__ import annotations

from typing import NoReturn

from axon.dtype import DType
from axon.tensor import Tensor

__all__ = [
  "abs",
  "add",
  "cos",
  "div",
  "exp",
  "flatten",
  "gather",
  "identity",
  "inv",
  "log",
  "matmul",
  "max",
  "maximum",
  "mean",
  "mul",
  "neg",
  "pow",
  "reshape",
  "sin",
  "sqrt",
  "squeeze",
  "sub",
  "sum",
  "transpose",
  "unsqueeze",
  "where",
]


def _not_implemented(name: str) -> NoReturn:
  raise NotImplementedError(f"axon.functional.{name} is not implemented yet.")


def add[D: DType](a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
  from axon.operation.binary import Add

  return Add[D]()(a, b)


def mul[D: DType](a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
  from axon.operation.binary import Mul

  return Mul[D]()(a, b)


def matmul[D: DType](a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
  from axon.operation.binary import MatMul

  return MatMul[D]()(a, b)


def pow[D: DType](a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
  from axon.operation.binary import Pow

  return Pow[D]()(a, b)


def sub(a: Tensor, b: Tensor) -> Tensor:
  _ = (a, b)
  _not_implemented("sub")


def div(a: Tensor, b: Tensor) -> Tensor:
  _ = (a, b)
  _not_implemented("div")


def identity(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("identity")


def neg(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("neg")


def inv(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("inv")


def reshape(x: Tensor, shape: tuple[int, ...]) -> Tensor:
  _ = (x, shape)
  _not_implemented("reshape")


def transpose(x: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
  _ = (x, axes)
  _not_implemented("transpose")


def squeeze(x: Tensor, axis: int | tuple[int, ...] | None = None) -> Tensor:
  _ = (x, axis)
  _not_implemented("squeeze")


def unsqueeze(x: Tensor, axis: int) -> Tensor:
  _ = (x, axis)
  _not_implemented("unsqueeze")


def flatten(x: Tensor, start_dim: int = 0) -> Tensor:
  _ = (x, start_dim)
  _not_implemented("flatten")


def sum(
  x: Tensor,
  axis: int | tuple[int, ...] | None = None,
  *,
  keepdims: bool = False,
) -> Tensor:
  _ = (x, axis, keepdims)
  _not_implemented("sum")


def mean(
  x: Tensor,
  axis: int | tuple[int, ...] | None = None,
  *,
  keepdims: bool = False,
) -> Tensor:
  _ = (x, axis, keepdims)
  _not_implemented("mean")


def max(
  x: Tensor,
  axis: int | tuple[int, ...] | None = None,
  *,
  keepdims: bool = False,
) -> Tensor:
  _ = (x, axis, keepdims)
  _not_implemented("max")


def gather(x: Tensor, indices: Tensor, axis: int) -> Tensor:
  _ = (x, indices, axis)
  _not_implemented("gather")


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
  _ = (condition, x, y)
  _not_implemented("where")


def exp(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("exp")


def log(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("log")


def sqrt(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("sqrt")


def abs(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("abs")


def sin(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("sin")


def cos(x: Tensor) -> Tensor:
  _ = x
  _not_implemented("cos")


def maximum(a: Tensor, b: Tensor) -> Tensor:
  _ = (a, b)
  _not_implemented("maximum")
