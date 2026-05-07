from __future__ import annotations

from typing import Any, cast

from axon.backend import xp
from axon.dtype import DType, promote
from axon.scalar import Scalar
from axon.var import Constant, Var, Node

__all__ = [
  "add",
  "div",
  "matmul",
  "mul",
  "neg",
  "pow",
  "sub",
]


def _to_node_pair[D: DType](
  a: Node[D] | Scalar, b: Node[D] | Scalar
) -> tuple[Node[D], Node[D]]:
  """Scalar 는 0-D Constant 로 lift, dtype 은 numpy-style promotion 으로 통일."""
  target = promote(
    a.dtype if isinstance(a, Node) else a,
    b.dtype if isinstance(b, Node) else b,
  )
  return cast(
    tuple[Node[D], Node[D]],
    (_coerce(a, target), _coerce(b, target)),
  )


def _coerce(x: Node[Any] | Scalar, target: type[DType]) -> Node:
  """target dtype 의 Node 로 강제. Var 의 dtype 불일치는 NotImplementedError
  (Cast Op 미구현)."""
  if isinstance(x, Node):
    if x.dtype is target:
      return x
    if isinstance(x, Var):
      raise NotImplementedError(
        f"dtype 불일치 cast 미구현. "
        f"Var.dtype={x.dtype.__name__}, target={target.__name__}."
      )
    return Constant(xp.array(cast(Any, x._data), dtype=target))
  return Constant(x, dtype=target)


def add[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  from axon.operation import Add

  a, b = _to_node_pair(a, b)
  return Add[D]().apply(a, b)


def mul[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  from axon.operation import Mul

  a, b = _to_node_pair(a, b)
  return Mul[D]().apply(a, b)


def matmul[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  from axon.operation import MatMul

  a, b = _to_node_pair(a, b)
  return MatMul[D]().apply(a, b)


def pow[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  from axon.operation import Pow

  a, b = _to_node_pair(a, b)
  return Pow[D]().apply(a, b)


def div[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  from axon.operation import Div

  a, b = _to_node_pair(a, b)
  return Div[D]().apply(a, b)


def neg[D: DType](a: Node[D] | Scalar) -> Node[D]:
  from axon.operation import Neg

  a = _to_node_pair(a, 0)[0] if not isinstance(a, Node) else a
  return Neg[D]().apply(a)


def sub[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  a, b = _to_node_pair(a, b)
  return add(a, neg(b))
