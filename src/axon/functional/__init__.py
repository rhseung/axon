"""연산 API.

현재 구현 완료된 primitive는 `add`, `mul`, `matmul`, `pow`만 지원한다.
나머지 API는 시그니처만 유지하고 `NotImplementedError`를 던진다.
"""

from __future__ import annotations

from typing import NoReturn

from numpy.typing import DTypeLike

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


def add[D: DTypeLike](a: Tensor[D], b: Tensor[D]):
  from axon.operation.binary import Add

  return Add()(a, b)


def mul[D: DTypeLike](a: Tensor[D], b: Tensor[D]):
  from axon.operation.binary import Mul

  return Mul()(a, b)


def matmul[D: DTypeLike](a: Tensor[D], b: Tensor[D]):
  from axon.operation.binary import MatMul

  return MatMul()(a, b)


def pow[D: DTypeLike](a: Tensor[D], b: Tensor[D]):
  from axon.operation.binary import Pow

  return Pow()(a, b)


def sub[D: DTypeLike](a: Tensor[D], b: Tensor[D]):
  _ = (a, b)
  _not_implemented("sub")


def div[D: DTypeLike](a: Tensor[D], b: Tensor[D]):
  _ = (a, b)
  _not_implemented("div")


def identity[D: DTypeLike](x: Tensor[D]):
  _ = x
  _not_implemented("identity")


def neg[D: DTypeLike](x: Tensor[D]):
  _ = x
  _not_implemented("neg")


def inv[D: DTypeLike](x: Tensor[D]):
  _ = x
  _not_implemented("inv")


def reshape[D: DTypeLike](x: Tensor[D], shape: tuple[int, ...]):
  _ = (x, shape)
  _not_implemented("reshape")


def transpose[D: DTypeLike](x: Tensor[D], axes: tuple[int, ...] | None = None):
  _ = (x, axes)
  _not_implemented("transpose")


def squeeze[D: DTypeLike](x: Tensor[D], axis: int | tuple[int, ...] | None = None):
  _ = (x, axis)
  _not_implemented("squeeze")


def unsqueeze[D: DTypeLike](x: Tensor[D], axis: int):
  _ = (x, axis)
  _not_implemented("unsqueeze")


def flatten[D: DTypeLike](x: Tensor[D], start_dim: int = 0):
  _ = (x, start_dim)
  _not_implemented("flatten")


def sum[D: DTypeLike](  # noqa: A001
  x: Tensor[D],
  axis: int | tuple[int, ...] | None = None,
  *,
  keepdims: bool = False,
):
  _ = (x, axis, keepdims)
  _not_implemented("sum")


def mean[D: DTypeLike](
  x: Tensor[D],
  axis: int | tuple[int, ...] | None = None,
  *,
  keepdims: bool = False,
):
  _ = (x, axis, keepdims)
  _not_implemented("mean")


def max[D: DTypeLike](  # noqa: A001
  x: Tensor[D],
  axis: int | tuple[int, ...] | None = None,
  *,
  keepdims: bool = False,
):
  _ = (x, axis, keepdims)
  _not_implemented("max")


def gather[D: DTypeLike](x: Tensor[D], indices: Tensor[D], axis: int):
  _ = (x, indices, axis)
  _not_implemented("gather")


def where[D: DTypeLike](condition: Tensor[D], x: Tensor[D], y: Tensor[D]):
  _ = (condition, x, y)
  _not_implemented("where")


def exp[D: DTypeLike](x: Tensor[D]) -> Tensor[D]:
  _ = x
  _not_implemented("exp")


def log[D: DTypeLike](x: Tensor[D]) -> Tensor[D]:
  _ = x
  _not_implemented("log")


def sqrt[D: DTypeLike](x: Tensor[D]) -> Tensor[D]:
  _ = x
  _not_implemented("sqrt")


def abs[D: DTypeLike](x: Tensor[D]) -> Tensor[D]:  # noqa: A001
  _ = x
  _not_implemented("abs")


def sin[D: DTypeLike](x: Tensor[D]) -> Tensor[D]:
  _ = x
  _not_implemented("sin")


def cos[D: DTypeLike](x: Tensor[D]) -> Tensor[D]:
  _ = x
  _not_implemented("cos")


def maximum[D: DTypeLike](a: Tensor[D], b: Tensor[D]) -> Tensor[D]:
  _ = (a, b)
  _not_implemented("maximum")
