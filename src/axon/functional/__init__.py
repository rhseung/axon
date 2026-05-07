from __future__ import annotations

from typing import Any, cast

from axon.backend import xp
from axon.dtype import DType, promote
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


def _to_node_pair[D: DType](
  a: Node[D] | Scalar, b: Node[D] | Scalar
) -> tuple[Node[D], Node[D]]:
  """이항 입력 정규화: numpy-style type promotion 으로 target dtype 결정 후 양쪽
  모두 그 dtype 으로 강제 — Scalar 는 0-D non-grad Node 로 wrap, 이미 Node 면
  필요 시 cast.

  promotion 룰 (`axon.dtype.promote`):
  - Python `int` → `INT64`, `float` → `FLOAT32`, `bool` → `BOOL` 매핑
  - rank 가 가장 높은 dtype 이 target
  - 예: `Node[INT32] + 0.5` → target=`FLOAT32` (int → 0.5 가 float 라서)

  cast 정책:
  - dtype 일치: 그대로 사용
  - 불일치 + non-grad Node: `xp.array` 로 안전하게 재생성 (graph 추적 없으니
    재생성으로 무관)
  - 불일치 + grad Node: `NotImplementedError` — Cast Op 미구현이라 graph 보존
    못 함. 사용자가 명시적 cast 필요 (드물지만 `int_param + 0.5` 같은 케이스)

  0-D wrap 의 의미: `BinaryOp._validate` 의 글로벌 0-D 정책으로 메모리 오버헤드
  없이 backend broadcast. wrap 된 scalar Node 는 `_requires_grad=False` 라
  backward 의 needs_grad mask 가 해당 미분 항 자동 skip (음수 base + 상수 지수
  같은 케이스에서 NaN 회피).

  Scalar 받는 functional 함수 전반의 공통 dispatch — `pow` / `add` / `mul` /
  `div` 등 모두 같은 헬퍼로 처리.
  """
  target = promote(
    a.dtype if isinstance(a, Node) else a,
    b.dtype if isinstance(b, Node) else b,
  )

  return cast(
    tuple[Node[D], Node[D]],
    (_coerce(a, target), _coerce(b, target)),
  )


def _coerce(x: Node[Any] | Scalar, target: type[DType]) -> Node:
  """입력을 `target` dtype 의 Node 로 강제. dtype 이미 맞으면 그대로, Scalar 면
  0-D Node 로 wrap, 다른 dtype 의 grad Node 는 `NotImplementedError`.

  주의: promotion 으로 결과 dtype 이 호출자의 `[D: DType]` 와 달라질 수 있음
  (예: `add[INT32](node_int32, 0.5) → Node[FLOAT32]`). 정적 타입은 부정확하지만
  실제 사용 (mostly same-dtype) 에선 일치하므로 cast 로 처리. mixed-dtype 본격
  지원 시 BinaryOp 의 `[D]` 자체를 재설계해야 함.
  """
  if isinstance(x, Node):
    if x.dtype is target:
      return x
    if x.requires_grad:
      raise NotImplementedError(
        f"dtype 불일치 cast 미구현 (Cast Op 부재). "
        f"node.dtype={x.dtype.__name__}, target={target.__name__}. "
        f"명시적 cast 필요."
      )
    # non-grad Node — graph 추적 없으니 재생성으로 cast
    return Node.from_array(xp.array(x._data, dtype=target))
  return Node(x, dtype=target)


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
  """거듭제곱 a ** b. Scalar 입력은 0-D non-grad Node 로 wrap 후 단일 `Pow` Op
  에 위임.

  0-D wrap 가능한 이유: `BinaryOp._validate` 가 한쪽이 0-D 인 경우를 모든
  BinaryOp 공통 정책으로 통과시킨다 (0-D 는 ambiguous broadcast 가 아니라
  단일 값). 메모리 오버헤드 없이 backend 가 자연스럽게 broadcast.

  안전성: wrap 된 scalar Node 는 `_requires_grad=False` (Node.from_array 기본).
  `Pow.backward_binary` 의 `needs_grad[1]` 가 False → log 항 skip → 음수 base
  + 상수 지수 같은 케이스에서도 NaN 없음.
  """
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
