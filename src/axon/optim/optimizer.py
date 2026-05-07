from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from axon.var import Var


class Optimizer(ABC):
  """단일 `Var` 의 update 정책 + per-weight state. 한 인스턴스는 한 Var 를 위한
  거 — 공유하면 momentum / Adam state 가 섞임. `Net.optimizer` default 가 자동
  deepcopy 부착.
  """

  @abstractmethod
  def update(self, var: Var) -> None:
    """`var._data` 를 `var.grad` 로 in-place update."""
    ...
