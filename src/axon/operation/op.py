from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from axon.dtype import DType
from axon.errors import ShapeError
from axon.node import Node

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class Op[D: DType](ABC):
  """연산 그래프 노드의 미분 규칙. forward/backward 는 순수 Array 산수, apply 가
  Node wrapping 을 담당한다.

  서브클래스가 구현하는 `forward` / `backward` 는 backend Array 만 다룬다 — 그래프
  metadata 는 무관. `apply` 가 입력 Node 들을 unwrap → forward 호출 → 결과 Array
  를 Node 로 wrap 하고, 입력 중 하나라도 `_requires_grad` 면 그래프 metadata 를
  세팅한다.

  주의: `forward` / `backward` 내부에서 Node 의 연산자 (`-x`, `x + y`, ...) 는
  쓰면 안 된다. Node 연산자는 다시 `Op.apply` 로 돌아오므로 무한 재귀.
  Array 의 연산자 (`a + b`) 는 backend native 산수라 OK.

  ## Shape precheck

  axon 은 broadcasting 을 허용하지 않는다. backend 가 silent 하게 broadcast 해
  버리기 전에 `apply` 가 forward 직전에 `_validate` 를 호출해 입력 shape 를
  검증한다 (Node → shape 만 추출). 위반 시 `ShapeError`. 기본은 no-op,
  `UnaryOp` / `BinaryOp` 가 각자 의미에 맞게 override 한다.

  ## needs_grad mask

  `backward` 는 `needs_grad: tuple[bool, ...]` 를 받아 어떤 입력에 대해서만
  ∂L/∂x 를 계산할지 안다. 상수처럼 추적 불필요한 입력 (`_requires_grad=False`)
  슬롯에 대해서는 Op 가 `None` 을 반환해 계산을 건너뛰면 된다 — `Pow.log(a)` 처럼
  값이 NaN 위험이 있는 항을 안전하게 우회.
  """

  @abstractmethod
  def forward(self, *inputs: Array[D]) -> Array[D]:
    """순전파 y = f(x_1, ..., x_n)."""
    ...

  @abstractmethod
  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
    needs_grad: tuple[bool, ...],
  ) -> tuple[Array[D] | None, ...]:
    """체인룰로 입력별 손실 편미분 (∂L/∂x_1, ..., ∂L/∂x_n).
    `needs_grad[i] == False` 슬롯은 `None` 반환으로 계산 생략 가능.
    """
    ...

  @abstractmethod
  def _validate(self, *input_shapes: tuple[int, ...]) -> None:
    """입력 shape precheck — `apply` 가 forward 직전에 호출.

    arity 별 정책의 진입점이라 abstract — `UnaryOp` / `BinaryOp` 가 각각의 의미
    (단항 hook / 0-D 글로벌 정책 + 이항 검증) 를 정의한다. 새 arity (예:
    Ternary) 추가 시 직접 구현 필수. broadcasting 미지원 정책을 강제해 backend
    별로 다른 silent broadcast 동작이 생기지 않도록 하는 게 목적이다.
    """
    ...

  def apply(self, *inputs: Node[D]) -> Node[D]:
    self._validate(*(n.shape for n in inputs))
    out_array = self.forward(*(n._data for n in inputs))
    out = Node.from_array(out_array)
    if any(n.requires_grad for n in inputs):
      out._op = self
      out._inputs = cast(tuple[Node, ...], inputs)
      out._requires_grad = True
    return out


class UnaryOp[D: DType](Op[D]):
  """단항 연산 기본 클래스.

  `validate_shape(shape)` 는 서브클래스 전용 hook — 단항 Op 은 inter-input 비교가
  없으니 default 는 no-op, `Reshape` 처럼 `prod` 일치 등을 강제할 때 override.

  `backward_unary` 는 `needs_grad` 를 받지 않는다 — UnaryOp 의 backward 가 호출
  되었다는 건 유일한 입력이 `requires_grad` 인 상태이기 때문 (`Op.apply` 의
  자동 전파 규칙: 출력의 `_requires_grad` 는 입력 중 하나라도 True 일 때만 True).
  """

  @abstractmethod
  def forward_unary(self, x: Array[D]) -> Array[D]:
    """순전파 y = f(x)."""
    ...

  @abstractmethod
  def backward_unary(self, grad: Array[D], x: Array[D]) -> Array[D]:
    """역전파 ∂L/∂x."""
    ...

  def validate_shape(self, shape: tuple[int, ...]) -> None:
    """입력 shape 검증 hook. 기본 no-op — 서브클래스가 필요 시 override 해
    `ShapeError` 를 던진다 (예: `Reshape` 가 `prod(in) == prod(out)` 강제).
    """
    pass

  def forward(self, *inputs: Array[D]) -> Array[D]:
    (x,) = inputs
    return self.forward_unary(x)

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
    needs_grad: tuple[bool, ...],
  ) -> tuple[Array[D] | None, ...]:
    # `needs_grad` 는 시그니처 일관성으로만 받음. UnaryOp 은 호출 시점에 (True,)
    # 가 보장되므로 (Op.apply 의 _requires_grad 자동 전파 규칙 — 출력이 True 면
    # 유일한 입력도 True) backward_unary 분기에 mask 가 필요 없다.
    (x,) = inputs
    return (self.backward_unary(grad, x),)

  def _validate(self, *input_shapes: tuple[int, ...]) -> None:
    (shape,) = input_shapes
    self.validate_shape(shape)


class BinaryOp[D: DType](Op[D]):
  """이항 연산: `forward_binary(a, b)`, `backward_binary(grad, a, b, needs_grad)` —
  `forward`/`backward` 는 `*inputs` 로 위임.

  `validate_shape(a, b)` 의 기본 구현은 elementwise 가정으로 `a == b` 를 강제
  (broadcasting 미지원). `MatMul` 등 다른 shape 규칙을 가진 Op 은 `validate_shape`
  를 override.

  `backward_binary` 는 `needs_grad: tuple[bool, bool]` 을 받아, 한 쪽 입력이
  상수 (Scalar 를 wrap 한 non-grad Node) 일 때 해당 미분 항을 건너뛸 수 있다.
  사용 예: `Pow` 가 `b` 가 상수일 때 `xp.log(a)` 항을 skip → 음수 base 에서
  NaN 발생 차단. 건너뛴 슬롯은 `None` 반환.
  """

  @abstractmethod
  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """순전파 y = f(a, b)."""
    ...

  @abstractmethod
  def backward_binary(
    self,
    grad: Array[D],
    a: Array[D],
    b: Array[D],
    *,
    needs_grad: tuple[bool, bool],
  ) -> tuple[Array[D] | None, Array[D] | None]:
    """역전파 (∂L/∂a, ∂L/∂b). `needs_grad[i] == False` 슬롯은 None 반환 가능."""
    ...

  def validate_shape(self, a: tuple[int, ...], b: tuple[int, ...]) -> None:
    """이항 연산 입력 shape 검증. 기본은 elementwise — `a == b` 강제.
    matmul 처럼 다른 규칙을 가진 Op 은 override.

    호출 시점에 한쪽이 0-D 인 경우는 이미 `_validate` 에서 걸러졌다 — 서브클래스
    override 는 multi-dim 케이스만 신경 쓰면 됨.
    """
    if a != b:
      raise ShapeError(
        f"{type(self).__name__}: 입력 shape 불일치, a.shape={a}, b.shape={b}"
      )

  def forward(self, *inputs: Array[D]) -> Array[D]:
    a, b = inputs
    return self.forward_binary(a, b)

  def backward(
    self,
    grad: Array[D],
    *inputs: Array[D],
    needs_grad: tuple[bool, ...],
  ) -> tuple[Array[D] | None, ...]:
    a, b = inputs
    mask = cast(tuple[bool, bool], needs_grad)
    return self.backward_binary(grad, a, b, needs_grad=mask)

  def _validate(self, *input_shapes: tuple[int, ...]) -> None:
    """글로벌 0-D scalar 정책: 한쪽이라도 0-D 면 통과.

    no-broadcast 정책의 진짜 목적은 `(3,)` vs `(3,1)` 같은 ambiguous shape
    변환 차단이고, 0-D 는 broadcasting 되어도 의미상 단일 값이라 ambiguity 가
    없다. 이 체크를 dispatcher 에 두면 모든 BinaryOp 서브클래스 (Add/Mul/Div/
    Pow/MatMul ...) 가 자동으로 동일한 0-D 정책을 따르고, `validate_shape` 는
    multi-dim 규칙만 정의하면 된다.

    글로벌 정책을 무시하려면 서브클래스가 `_validate` 자체를 override.
    """
    a, b = input_shapes
    if a == () or b == ():
      return
    self.validate_shape(a, b)
