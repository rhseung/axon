from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from axon.backend import xp
from axon.backend._dtype import from_backend_dtype
from axon.backend.protocol import Array
from axon.dtype import DType

if TYPE_CHECKING:
  from axon.operation.op import Op
  from axon.optim.optimizer import Optimizer


class Node[D: DType = DType](ABC):
  """Constant / Var 의 공통 base. `_from_op` 가 abstract 라 직접 인스턴스화 불가."""

  _data: Array[D]
  _op: Op | None
  _inputs: tuple[Node, ...]

  def __init__(self, data: Any, *, dtype: type[DType] = DType.FLOAT32):
    self._data = cast(Array[D], xp.array(data, dtype=dtype))
    self._op = None
    self._inputs = ()

  @classmethod
  @abstractmethod
  def _from_op[D2: DType](
    cls,
    data: Array[D2],
    op: Op,
    inputs: tuple[Node, ...],
  ) -> Node[D2]: ...

  @property
  def dtype(self) -> type[DType]:
    return from_backend_dtype(self._data.dtype)

  @property
  def shape(self) -> tuple[int, ...]:
    return self._data.shape

  @property
  def ndim(self) -> int:
    return self._data.ndim

  @property
  def op(self) -> Op | None:
    return self._op

  @property
  def inputs(self) -> tuple[Node, ...]:
    return self._inputs

  def as_numpy(self) -> Any:
    return xp.to_numpy(cast(Array, self._data))

  def __repr__(self) -> str:
    cls = type(self).__name__
    return f"{cls}(shape={self.shape}, dtype={self.dtype.__name__})"

  def __add__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import add

    return add(self, other)

  def __radd__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import add

    return add(other, self)

  def __sub__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import sub

    return sub(self, other)

  def __rsub__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import sub

    return sub(other, self)

  def __mul__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import mul

    return mul(self, other)

  def __rmul__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import mul

    return mul(other, self)

  def __truediv__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import div

    return div(self, other)

  def __rtruediv__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import div

    return div(other, self)

  def __pow__(self, other: Node[D] | int | float) -> Node[D]:
    from axon.functional import pow as node_pow

    return node_pow(self, other)

  def __rpow__(self, other: int | float) -> Node[D]:
    from axon.functional import pow as node_pow

    return node_pow(other, self)

  def __matmul__(self, other: Node[D]) -> Node[D]:
    from axon.functional import matmul

    return matmul(self, other)

  def __rmatmul__(self, other: Node[D]) -> Node[D]:
    from axon.functional import matmul

    return matmul(other, self)

  def __neg__(self) -> Node[D]:
    from axon.functional import neg

    return neg(self)


class Constant[D: DType = DType](Node[D]):
  """비추적 노드 — 입력 데이터, 비학습 상수, KV cache."""

  @classmethod
  def _from_op[D2: DType](
    cls,
    data: Array[D2],
    op: Op,
    inputs: tuple[Node, ...],
  ) -> Constant[D2]:
    out = cast(Constant[D2], object.__new__(cls))
    out._data = data
    out._op = op
    out._inputs = inputs
    return out


class Var[D: DType = DType](Node[D]):
  """추적 노드 — 학습 weight (leaf) 또는 추적 중간 결과."""

  grad: Array[D]
  _optimizer: Optimizer | None

  def __init__(
    self,
    data: Any,
    *,
    dtype: type[DType] = DType.FLOAT32,
    optimizer: Optimizer | None = None,
  ):
    super().__init__(data, dtype=dtype)
    self.grad = cast(Array[D], xp.zeros_like(cast(Array, self._data)))
    self._optimizer = optimizer

  @classmethod
  def _from_op[D2: DType](
    cls,
    data: Array[D2],
    op: Op,
    inputs: tuple[Node, ...],
  ) -> Var[D2]:
    out = cast(Var[D2], object.__new__(cls))
    out._data = data
    out._op = op
    out._inputs = inputs
    out.grad = cast(Array[D2], xp.zeros_like(cast(Array, data)))
    out._optimizer = None
    return out

  @property
  def is_parameter(self) -> bool:
    """학습 leaf — `_op is None` 이면 leaf."""
    return self._op is None

  def backward(self) -> None:
    """그래프 따라 grad 누적. zero 는 `optimize()` / `model.zero_grad()` 가 처리."""
    self.grad = cast(Array[D], xp.ones_like(cast(Array, self._data)))

    for n in _topological_order(self):
      if not isinstance(n, Var) or n._op is None:
        continue
      inputs = tuple(inp._data for inp in n._inputs)
      input_grads = n._op.backward(n.grad, *inputs)
      assert len(input_grads) == len(n._inputs), (
        f"{type(n._op).__name__}.backward 가 입력 {len(n._inputs)} 개에 대해 "
        f"grad {len(input_grads)} 개 반환 — 길이 불일치."
      )
      for inp, inp_grad in zip(n._inputs, input_grads):
        if isinstance(inp, Var):
          inp.grad += inp_grad

  def optimize(self) -> None:
    """그래프 도달 가능한 학습 weight 의 update + grad zero."""
    for n in _topological_order(self):
      if isinstance(n, Var) and n.is_parameter and n._optimizer is not None:
        n._optimizer.update(n)
        n.grad = xp.zeros_like(n._data)


def _topological_order(root: Node[Any]) -> list[Node[Any]]:
  order: list[Node[Any]] = []
  visited: set[int] = set()

  def visit(node: Node[Any]) -> None:
    visited.add(id(node))
    for inp in node._inputs:
      if id(inp) not in visited:
        visit(inp)
    order.append(node)

  visit(root)
  return order[::-1]
