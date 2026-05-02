from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
  from axon.op import Op, Identity

"""
examples/pure_mnist.py에서 직접 구현할 결과를 바탕으로 추상화를 진행한다.
이 때, 직접 구현해보면서 최적화 및 추상화가 느껴진 점은, 각 레이어마다 역전파를 정의하지 말고, 덧셈, 곱셈, 행렬 곱 등의 primitive operation에 대해서만 역전파를 정의하면,
cross entropy loss 같은 레이어의 역전파를 수학적으로 계산하여 기술하지 않아도 자동 미분이 가능하다는 것이다.

그래서, Op 라는 추상 클래스를 선언하여 primitive operation들의 순전파와 역전파를 구현해준다. 이 때 각 Op 들의 backward 메소드는 역전파 과정 중의 grad 값을 넣어주게 된다.
각 텐서는 _op, _inputs 필드를 가지게 된다. 이는 해당 텐서가 _inputs와 _op로 연산되어 생성된 텐서란 뜻으로, _inputs 또한 텐서들의 집합이므로 이는 연산 그래프를 나타내기도 한다.
따라서 텐서의 역전파를 구현할 때는 파라미터 의존 문제가 발생하지 않도록 위상 정렬을 한 뒤, 차례대로 inp에 대해 op backward를 계산하게 해준다.
이 때, 텐서의 역전파에 들어가는 grad 인자는 역시 op에 전달될, 역전파 과정 중의 grad 값으로, 가장 upstream의 grad는 ∂loss/∂loss = 1이므로 np.ones_like를 사용한다.

그렇다면 텐서에 속해있는 grad 속성은 왜 필요한걸까? 이는 optimizer가 가중치의 그라디언트를 바탕으로 업데이트할 때, 기존 pure 코드에서는 W, b, dW, db 속성을 따로 저장하고 있다.
이를 axon 코드에서는 텐서에 grad 속성을 추가해서 텐서가 본인의 그라디언트와 값을 동시에 소유하게 하는 것이다.

이제 앞으로 할 일은 primitive 연산들 잘 선언하고, 연산자 오버로딩 잘 매핑하고, numpy 유틸도 텐서에서 쓸 수 있게 잘 지원하는 것.
"""


class Tensor[D: DTypeLike]:
  def __init__(self, data: NDArray[Any], *, dtype: DTypeLike = np.float32):
    self.data = np.asarray(data, dtype=dtype)
    self.grad = np.zeros_like(self.data)

    # 이 Tensor는 _inputs, _op의 조합으로 생성되었습니다.
    self._op: Op = Identity()
    self._inputs: tuple[Tensor, ...] = ()

  def backward(self, grad: Tensor | None = None):
    if grad is None:
      # 역전파의 가장 upstream 값은 ∂loss/∂loss = 1로 지정.
      grad = Tensor(np.ones_like(self.data))
    self.grad = grad

    topological_inputs = []
    visited = set()

    def build(t: Tensor):
      visited.add(t)
      for inp in t._inputs:
        if id(inp) not in visited:
          build(inp)

      topological_inputs.append(t)

    build(self)

    for inp in reversed(topological_inputs):
      inp._op.backward(inp.grad, inp._inputs)
