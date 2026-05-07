from __future__ import annotations

from typing import TYPE_CHECKING, cast

from axon.backend import xp
from axon.dtype import DType
from axon.operation.op import Op

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class CrossEntropy[D: DType](Op[D]):
  """fused softmax + NLL. log-sum-exp shift 로 overflow 회피, closed-form
  `(softmax − one_hot)/N` backward. target 은 학습 대상이 아니라 Op 인스턴스
  필드로 보관 (mixed dtype 회피).
  """

  def __init__(self, target: Array, reduction: str = "mean"):
    if reduction not in ("mean", "sum", "none"):
      raise ValueError(
        f"reduction 은 mean/sum/none 중 하나여야 합니다. got {reduction!r}"
      )
    self.target = target
    self.reduction = reduction

  def forward(self, *inputs: Array[D]) -> Array[D]:
    """loss = -log_softmax(logits)[range(N), target], reduction 적용."""
    (logits,) = inputs
    arr = cast("Array", logits)
    log_softmax = self._log_softmax(arr)
    one_hot = self._one_hot(logits.shape[-1])
    nll = -xp.sum(log_softmax * one_hot, axis=-1)  # (N,)

    if self.reduction == "mean":
      return cast("Array[D]", xp.mean(nll))
    if self.reduction == "sum":
      return cast("Array[D]", xp.sum(nll))
    return cast("Array[D]", nll)

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
  ) -> tuple[Array[D], ...]:
    """∂loss/∂logits = (softmax − one_hot) · scale.
    scale: mean → grad/N, sum → grad, none → grad[:, None]."""
    (logits,) = inputs
    arr = cast("Array", logits)
    g = cast("Array", grad)
    n, k = logits.shape[0], logits.shape[-1]

    softmax = self._softmax(arr)
    one_hot = self._one_hot(k)
    diff = softmax - one_hot  # (N, K)

    if self.reduction == "mean":
      scale = g / n
    elif self.reduction == "sum":
      scale = g
    else:  # "none"
      scale = xp.expand_dims(g, axis=-1)

    return (cast("Array[D]", scale * diff),)

  def _log_softmax(self, arr: Array) -> Array:
    max_logits = xp.max(arr, axis=-1, keepdims=True)
    shifted = arr - max_logits
    log_sum_exp = xp.log(xp.sum(xp.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp

  def _softmax(self, arr: Array) -> Array:
    max_logits = xp.max(arr, axis=-1, keepdims=True)
    exp_shifted = xp.exp(arr - max_logits)
    return exp_shifted / xp.sum(exp_shifted, axis=-1, keepdims=True)

  def _one_hot(self, num_classes: int) -> Array:
    return xp.take(xp.eye(num_classes), self.target, axis=0)
