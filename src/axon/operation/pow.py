from __future__ import annotations

from typing import cast

from axon.backend import xp
from axon.backend.protocol import Array
from axon.dtype import DType
from axon.operation.op import BinaryOp


class Pow[D: DType](BinaryOp[D]):
  """거듭제곱 y = a ** b. 두 입력 모두 Node — 상수 지수/밑은 `functional.pow` 가
  0-D non-grad Node 로 wrap 해 일반화 경로로 보낸다.

  ## 음수 base 안전성

  ∂y/∂b = a ** b * log(a) 항은 a < 0 에서 NaN. 하지만 b 가 상수 (`requires_grad=
  False`) 이면 이 항을 *애초에 계산하지 않아도* 되므로, `backward_binary` 가
  `needs_grad[1]` 을 보고 log 계산 자체를 skip → `(-3.0) ** 2` 같은 표현식의
  grad 가 안전하게 계산된다.
  """

  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """거듭제곱 순전파 y = a ** b (원소별) 를 계산한다."""
    return a**b

  def backward_binary(
    self,
    grad: Array[D],
    a: Array[D],
    b: Array[D],
    *,
    needs_grad: tuple[bool, bool],
  ) -> tuple[Array[D] | None, Array[D] | None]:
    """y = a ** b 의 체인룰 (원소별).

    ∂y/∂a = b * a ** (b - 1), ∂y/∂b = a ** b * log(a).

    `needs_grad[1] == False` 이면 log(a) 계산을 건너뛴다 — a 에 음수가 있어도
    NaN 발생 없음 (상수 지수의 안전성).
    """
    da = grad * b * a ** (b - 1) if needs_grad[0] else None
    if needs_grad[1]:
      db = grad * a**b * cast(Array[D], xp.log(cast(Array, a)))
    else:
      db = None
    return (da, db)
