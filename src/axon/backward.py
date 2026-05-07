from __future__ import annotations

from typing import TYPE_CHECKING

from axon.backend import xp
from axon.node import Node
from axon.parameter import Parameter

if TYPE_CHECKING:
  from axon.backend.protocol import Array


def _topological_order(node: Node):
  ret: list[Node] = []
  visited: set[int] = set()

  def dfs(n: Node):
    visited.add(id(n))

    for n2 in n.inputs:
      if id(n2) not in visited:
        dfs(n2)

    ret.append(n)

  dfs(node)

  return reversed(ret)


def backward(loss: Node):
  grads: dict[int, Array] = {}
  grads[id(loss)] = xp.ones_like(loss._data)

  for n in _topological_order(loss):
    grad = grads.get(id(n))
    if grad is None:  # 위상 정렬하기에 n은 loss부터 나와서 괜찮음.
      continue

    if n.op is None:
      continue

    inputs = tuple(inp._data for inp in n.inputs)
    needs_grad = tuple(inp.requires_grad for inp in n.inputs)
    input_grads = n.op.backward(grad, *inputs, needs_grad=needs_grad)

    for inp, inp_grad in zip(n.inputs, input_grads):
      if not inp.requires_grad or inp_grad is None:
        continue

      grads[id(inp)] = grads.get(id(inp), xp.zeros_like(inp._data)) + inp_grad

      if isinstance(inp, Parameter):
        inp.grad += inp_grad
