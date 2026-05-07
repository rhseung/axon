from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from axon.backend import xp
from axon.var import Var

if TYPE_CHECKING:
  from axon.backend.protocol import Array
  from axon.optim.optimizer import Optimizer


def _walk_params(value: Any, visited: set[int] | None = None) -> Iterator[Var]:
  """value 안의 학습 leaf 를 yield. Net / dict / list / tuple 안까지 재귀."""
  if visited is None:
    visited = set()
  if id(value) in visited:
    return
  visited.add(id(value))

  if isinstance(value, Var):
    if value.is_parameter:
      yield value
    return
  if isinstance(value, Net):
    for child in value.__dict__.values():
      yield from _walk_params(child, visited)
  elif isinstance(value, dict):
    for child in value.values():
      yield from _walk_params(child, visited)
  elif isinstance(value, (list, tuple)):
    for child in value:
      yield from _walk_params(child, visited)


class Net(ABC):
  """학습 weight 를 들고 forward 를 정의하는 모듈.

  - `optimizer` class attribute 로 default 지정 → `__setattr__` 가 sub-module /
    list / dict 안의 미부착 weight 에 deepcopy 부착.
  - `parameters()` / `zero_grad` / `clip_grad_norm` / `grad_norm` 으로 글로벌 동작.
  - 호출은 `model.forward(x)` 명시 — `__call__` 안 두는 이유는 base 의
    `*args, **kwargs` 시그니처가 subclass forward 의 타입 강제를 우회하기 때문.
  """

  optimizer: Optimizer | None = None

  def __setattr__(self, name: str, value: Any) -> None:
    super().__setattr__(name, value)
    default_opt = type(self).optimizer
    if default_opt is None:
      return
    for p in _walk_params(value):
      if p._optimizer is None:
        p._optimizer = copy.deepcopy(default_opt)

  @abstractmethod
  def forward(self, *args: Any, **kwargs: Any) -> Var: ...

  def parameters(self) -> list[Var]:
    return list(_walk_params(self))

  def zero_grad(self) -> None:
    for p in self.parameters():
      p.grad = xp.zeros_like(p._data)

  def clip_grad_norm(self, max_norm: float) -> Array:
    """L2 norm 이 max_norm 넘으면 동일 비율로 축소. 반환은 clipping 전 norm."""
    params = self.parameters()
    total_norm_sq = sum((xp.sum(p.grad**2) for p in params), start=xp.array(0.0))
    total_norm = xp.sqrt(total_norm_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    if bool(clip_coef < 1):
      for p in params:
        p.grad = p.grad * clip_coef
    return total_norm

  def grad_norm(self) -> Array:
    params = self.parameters()
    total_norm_sq = sum((xp.sum(p.grad**2) for p in params), start=xp.array(0.0))
    return xp.sqrt(total_norm_sq)
