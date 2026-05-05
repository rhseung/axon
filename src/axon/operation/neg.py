from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.operation.op import UnaryOp

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Neg[D: DType](UnaryOp[D]):
  def forward_unary(self, x: Array[D]) -> Array[D]:
    """단항 부호 반전 순전파 y = -x 를 계산한다."""
    return -x

  def backward_unary(self, grad: Array[D], x: Array[D]) -> Array[D]:
    """y = -x 의 체인룰. ∂y/∂x = -1, grad 부호만 반전."""
    return -grad
