from __future__ import annotations

from typing import cast

import axon.functional as F
from axon.net.net import Net
from axon.var import Node, Var


class Sigmoid(Net):
  """`F.sigmoid` 의 thin wrapper."""

  def forward(self, x: Node) -> Var:
    return cast(Var, F.sigmoid(x))
