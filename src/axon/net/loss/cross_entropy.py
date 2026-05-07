from __future__ import annotations

from typing import Any, cast

import axon.functional as F
from axon.net.net import Net
from axon.var import Node, Var


class CrossEntropyLoss(Net):
  """fused softmax + NLL — `forward(logits, target)`. target 은 int 인덱스
  `Constant` 또는 raw int array.
  """

  def __init__(self, *, reduction: str = "mean"):
    self.reduction = reduction

  def forward(self, logits: Node, target: Node | Any) -> Var:
    return cast(Var, F.cross_entropy(logits, target, reduction=self.reduction))
