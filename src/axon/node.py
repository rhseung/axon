from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from axon.backend import xp
from axon.backend._dtype import from_backend_dtype
from axon.backend.protocol import Array
from axon.dtype import DType

if TYPE_CHECKING:
  from axon.operation.op import Op

"""
## 추상화의 동기

examples/pure_mnist.py 에서 직접 구현해본 결과를 바탕으로 한 추상화. 직접 구현해
보며 느낀 점: 각 레이어마다 역전파를 정의하지 말고, 덧셈/곱셈/행렬곱 같은 primitive
operation 에 대해서만 역전파를 정의하면, cross entropy loss 같은 레이어의 역전파를
수학적으로 계산하여 기술하지 않아도 자동 미분이 가능하다.

그래서 `Op` 라는 추상 클래스를 선언하여 primitive operation 들의 순전파/역전파를
구현해준다. 각 Op 의 backward 메소드는 역전파 과정 중의 grad 값을 받는다.
각 노드는 `_op`, `_inputs` 필드를 가진다 — 해당 노드가 `_inputs` 와 `_op` 로
연산되어 생성되었다는 뜻이고, `_inputs` 또한 노드들의 집합이므로 이는 연산 그래프를
나타낸다. 역전파 시에는 파라미터 의존 문제가 없도록 위상 정렬한 뒤 차례대로 inp 에
대해 op backward 를 계산. 가장 upstream 의 grad 는 ∂loss/∂loss = 1 이므로
xp.ones_like 사용.

`grad` 속성이 Parameter 에 속하는 이유: optimizer 가 가중치의 그라디언트를 바탕으로
업데이트할 때, 기존 pure 코드에서는 W, b, dW, db 속성을 따로 저장. axon 에서는
Parameter 에 grad 속성을 추가해서 가중치가 본인의 그라디언트와 값을 동시에 소유.

## 레이어 분해

- `Array` (`backend.protocol.Array`): backend-native ndarray (numpy/mlx/cupy).
  연산자, shape/dtype, Self 반환 등 numpy-like 인터페이스 모두 갖춤. "순수 데이터"
  역할을 단독으로 충족 — axon 에 별도 Tensor 클래스를 두지 않는 이유.
- `Node`: Array + 그래프 위치 (`_op`, `_inputs`, `_requires_grad`). 자동미분 단위.
- `Parameter(Node)`: Node + 영구 grad 버퍼. 학습 가능 leaf.

내부 storage `_data` 는 현재 활성 backend (`xp`) 가 만든 native ndarray —
`Array[D]` Protocol 만족. backend 전환은 `axon.set_backend(...)` 로 하고, Node 는
backend 를 직접 알지 않는다.

## `_requires_grad` 의 의미

"이 노드가 backward 그래프에 추적되어야 하는가" 를 표시한다.
- 일반 Node (사용자 입력 데이터, 상수): False.
- `Parameter`: True (학습 대상 leaf).
- Op 결과: 입력 중 하나라도 True 면 True 로 자동 전파 (`Op.apply` 참조).

"그러면 `isinstance(n, Parameter)` 로 대체하면 되지 않나?" 하는 오해가 있을 수
있는데, 그건 안 된다. `W` 가 Parameter 고 `x` 가 일반 Node 일 때 `x @ W` 는 학습
가능한 leaf 가 아니므로 의미상 Parameter 가 될 수 없지만 (Parameter 는 optimizer 가
업데이트하는 weight), W 까지 grad 가 흘러가려면 backward 그래프에는 포함되어야
한다. 즉 "Parameter 다" 와 "backward 추적이 필요하다" 는 일치하지 않는다 — 후자가
더 넓다. 그래서 `Op.apply` 가 중간 결과의 `_requires_grad` 를 True 로 자동 전파하고,
`_requires_grad` 필드 자체는 "Parameter 가 아니지만 추적은 필요한 중간 노드"
케이스를 표현하기 위해 존재한다.

## forward/backward 가 Array 만 다루는 이유

Op 의 `forward` / `backward` 는 순수 수학 (값 → 값) 이므로 그래프 metadata 를
알 필요가 없다. Op.apply 가 Node 를 unwrap → forward 호출 → 결과 Array 를 Node 로
wrap 하는 plumbing 을 한 곳에서 처리. 이 분리로 Op 구현체는 numpy 사용자가 읽을
수 있는 수식 코드 그대로가 된다 (`return a + b`).
"""


class Node[D: DType = DType]:
  """자동미분 그래프 노드. `_data` 가 값 (backend Array), 나머지는 그래프 metadata."""

  _data: Array[D]
  _op: Op | None
  _inputs: tuple[Node, ...]
  _requires_grad: bool

  def __init__(self, data: Any, *, dtype: type[DType] = DType.FLOAT32):
    self._data = cast(Array[D], xp.array(data, dtype=dtype))
    self._op = None
    self._inputs = ()
    self._requires_grad = False

  @staticmethod
  def from_array[D2: DType](data: Array[D2]) -> Node[D2]:
    """이미 backend 가 만든 Array 를 재변환 없이 Node 로 감싼다.

    Op.apply 가 forward 결과 Array 를 Node 로 올릴 때 사용. graph 추적 여부는
    호출자가 결정해 `_op`, `_inputs`, `_requires_grad` 를 세팅한다.
    """
    out = cast(Node[D2], object.__new__(Node))
    out._data = data
    out._op = None
    out._inputs = ()
    out._requires_grad = False
    return out

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
  def requires_grad(self) -> bool:
    return self._requires_grad

  @property
  def op(self) -> Op | None:
    return self._op

  @property
  def inputs(self) -> tuple[Node, ...]:
    return self._inputs

  def as_numpy(self) -> Any:
    return xp.to_numpy(cast(Array, self._data))

  def __add__(self, other: Node[D]) -> Node[D]:
    from axon.functional import add

    return add(self, other)

  def __radd__(self, other: Node[D]) -> Node[D]:
    from axon.functional import add

    return add(other, self)

  def __sub__(self, other: Node[D]) -> Node[D]:
    from axon.functional import sub

    return sub(self, other)

  def __rsub__(self, other: Node[D]) -> Node[D]:
    from axon.functional import sub

    return sub(other, self)

  def __mul__(self, other: Node[D]) -> Node[D]:
    from axon.functional import mul

    return mul(self, other)

  def __rmul__(self, other: Node[D]) -> Node[D]:
    from axon.functional import mul

    return mul(other, self)

  def __truediv__(self, other: Node[D]) -> Node[D]:
    from axon.functional import div

    return div(self, other)

  def __rtruediv__(self, other: Node[D]) -> Node[D]:
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
