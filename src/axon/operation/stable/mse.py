from __future__ import annotations

from typing import TYPE_CHECKING, cast

from axon.backend import xp
from axon.dtype import DType
from axon.operation.op import Op

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class MSE[D: DType](Op[D]):
  """`loss = reduction((pred − target)²)`.

  TODO: `Sum` / `Mean` reduction Op 도입 후 functional `mean(sub(pred,
  target) ** 2)` composition 으로 교체. stability 가 본질이 아니라 reduction
  Op 부재로 임시 fused.
  """

  def __init__(self, reduction: str = "mean"):
    if reduction not in ("mean", "sum", "none"):
      raise ValueError(
        f"reduction 은 mean/sum/none 중 하나여야 합니다. got {reduction!r}"
      )
    self.reduction = reduction

  def forward(self, *inputs: Array[D]) -> Array[D]:
    """y = reduction((pred - target)²)."""
    pred, target = inputs
    sq = (pred - target) * (pred - target)
    if self.reduction == "mean":
      return cast("Array[D]", xp.mean(cast("Array", sq)))
    if self.reduction == "sum":
      return cast("Array[D]", xp.sum(cast("Array", sq)))
    return sq

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂loss/∂pred = 2·(pred−target)·scale, ∂loss/∂target = −그것."""
    pred, target = inputs
    diff = pred - target

    if self.reduction == "mean":
      scale = grad * 2.0 / pred.size
    else:
      scale = grad * 2.0

    d_pred = scale * diff
    return (d_pred, -d_pred)
