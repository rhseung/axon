from __future__ import annotations

from axon.dtype import DType
from axon.node import Node
from axon.scalar import Scalar

__all__ = [
  "add",
  "div",
  "matmul",
  "mul",
  "neg",
  "pow",
  "sub",
]


def add[D: DType](a: Node[D], b: Node[D]) -> Node[D]:
  from axon.operation import Add

  return Add[D]().apply(a, b)


def mul[D: DType](a: Node[D], b: Node[D]) -> Node[D]:
  from axon.operation import Mul

  return Mul[D]().apply(a, b)


def matmul[D: DType](a: Node[D], b: Node[D]) -> Node[D]:
  from axon.operation import MatMul

  return MatMul[D]().apply(a, b)


def pow[D: DType](a: Node[D] | Scalar, b: Node[D] | Scalar) -> Node[D]:
  """거듭제곱 a ** b. 입력 타입에 따라 dispatch — 상수는 Op 인스턴스 필드로 보관:
  - 둘 다 Node: 일반 `Pow` (backward 에 log 항 — a > 0 강제)
  - 지수만 상수: `PowConstExp(n)` — log 항 없음, 음수 base 안전
  - 밑만 상수: `PowConstBase(c)` — c > 0 강제
  """
  from axon.operation import Pow, PowConstBase, PowConstExp

  if isinstance(b, Scalar):
    assert isinstance(a, Node)
    return PowConstExp[D](b).apply(a)
  if isinstance(a, Scalar):
    assert isinstance(b, Node)
    return PowConstBase[D](a).apply(b)
  return Pow[D]().apply(a, b)


def div[D: DType](a: Node[D], b: Node[D]) -> Node[D]:
  from axon.operation import Div

  return Div[D]().apply(a, b)


def neg[D: DType](a: Node[D]) -> Node[D]:
  from axon.operation import Neg

  return Neg[D]().apply(a)


def sub[D: DType](a: Node[D], b: Node[D]) -> Node[D]:
  return add(a, neg(b))
