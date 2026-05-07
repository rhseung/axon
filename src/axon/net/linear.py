from __future__ import annotations

from typing import TYPE_CHECKING, cast

import axon.functional as F
from axon.backend import xp
from axon.net.net import Net
from axon.var import Node, Var

if TYPE_CHECKING:
  from axon.optim.optimizer import Optimizer


class Linear(Net):
  """y = x @ W (+ b). Kaiming uniform 초기화."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    optimizer: Optimizer | None = None,
  ):
    self.in_features = in_features
    self.out_features = out_features

    bound = (1.0 / in_features) ** 0.5
    w_init = xp.random.uniform((in_features, out_features), low=-bound, high=bound)
    self.W: Var = Var(w_init, optimizer=optimizer)

    self.b: Var | None
    if bias:
      self.b = Var(xp.zeros((out_features,)), optimizer=optimizer)
    else:
      self.b = None

  def forward(self, x: Node) -> Var:
    out = F.matmul(x, self.W)
    if self.b is not None:
      out = F.add(out, self.b)
    return cast(Var, out)
