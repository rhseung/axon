from numpy.typing import DTypeLike

from axon.parameter import Parameter
from axon.tensor import Tensor


def _topological_order[D: DTypeLike](tensor: Tensor[D]):
  ret = []
  visited = set()

  def dfs(t: Tensor[D]):
    visited.add(id(t))

    for t2 in t._inputs:
      if id(t2) not in visited:
        dfs(t2)

    ret.append(t)

  dfs(tensor)

  return reversed(ret)


def backward[D: DTypeLike](loss: Tensor[D]):
  grads: dict[int, Tensor[D]] = {}
  grads[id(loss)] = Tensor.ones_like(loss)

  for t in _topological_order(loss):
    grad = grads.get(id(t))
    if grad is None:  # 위상 정렬하기에 t는 loss부터 나와서 괜찮음.
      continue

    input_grads = t._op.backward(grad, *t._inputs)

    for inp, inp_grad in zip(t._inputs, input_grads):
      grads[id(inp)] = grads.get(id(inp), Tensor.zeros_like(inp)) + inp_grad

      if isinstance(inp, Parameter):
        inp.grad += inp_grad
