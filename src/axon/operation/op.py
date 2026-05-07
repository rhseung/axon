from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from axon.dtype import DType
from axon.var import Constant, Node, Var

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Op[D: DType](ABC):
  """연산 그래프 노드의 미분 규칙. forward/backward 는 Array 산수, apply 가 노드
  wrapping 을 담당한다.

  주의: `forward` / `backward` 안에서 Node 연산자 (`-x`, `x + y`) 쓰면 무한 재귀
  (다시 Op.apply 로 돌아옴). Array 연산자 (`a + b`) 만 사용.
  """

  @abstractmethod
  def forward(self, *inputs: Array[D]) -> Array[D]: ...

  @abstractmethod
  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """체인룰. 반환 tuple 길이는 inputs 길이와 같아야 함 — Var.backward 가 검증."""
    ...

  def validate(self, *inputs: Node[D]) -> None:
    """입력 검증. 기본 no-op — 형상 규칙이 까다로운 Op (MatMul 등) 만 override."""
    pass

  def apply(self, *inputs: Node[D]) -> Node[D]:
    """입력에 Var 하나라도 있으면 결과 Var, 모두 Constant 면 결과 Constant."""
    self.validate(*inputs)
    out_array = self.forward(*(n._data for n in inputs))
    if any(isinstance(n, Var) for n in inputs):
      return Var._from_op(out_array, op=self, inputs=cast(tuple[Node, ...], inputs))
    return Constant._from_op(out_array, op=self, inputs=cast(tuple[Node, ...], inputs))
