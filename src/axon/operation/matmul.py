from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.errors import ShapeError
from axon.operation.op import Op
from axon.var import Node

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class MatMul[D: DType](Op[D]):
  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = a @ b."""
    a, b = inputs
    return a @ b

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂L/∂a = grad @ b.T, ∂L/∂b = a.T @ grad."""
    a, b = inputs
    return (grad @ b.T, a.T @ grad)

  def validate(self, *inputs: Node[D]) -> None:
    """ndim ≥ 2, contracting dim 일치, batch dim 정확히 일치 (broadcast 금지)."""
    a, b = inputs
    if a.ndim < 2 or b.ndim < 2:
      raise ShapeError(
        f"MatMul: 두 입력 모두 ndim >= 2 필요. "
        f"a.shape={a.shape}, b.shape={b.shape}"
      )
    if a.shape[-1] != b.shape[-2]:
      raise ShapeError(
        f"MatMul: 수축 차원 불일치. a.shape={a.shape}, b.shape={b.shape}"
      )
    if a.shape[:-2] != b.shape[:-2]:
      raise ShapeError(
        f"MatMul: 배치 차원 불일치. a.shape={a.shape}, b.shape={b.shape}"
      )
