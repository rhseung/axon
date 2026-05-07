from __future__ import annotations

from typing import cast

import axon.functional as F
from axon.net.net import Net
from axon.var import Node, Var


class MSELoss(Net):
  """`reduction((pred − target)²)` — `forward(pred, target)` 두 인자."""

  def __init__(self, *, reduction: str = "mean"):
    self.reduction = reduction

  def forward(self, pred: Node, target: Node) -> Var:
    return cast(Var, F.mse(pred, target, reduction=self.reduction))
