from __future__ import annotations

import math
from typing import cast

from axon.backend import xp
from axon.backend.protocol import Array
from axon.dtype import DType
from axon.operation.op import BinaryOp, UnaryOp
from axon.scalar import Scalar


class Pow[D: DType](BinaryOp[D]):
  """둘 다 Node 인 거듭제곱: y = a ** b. backward 에 `xp.log(a)` 항이 있어
  a 에 음수가 있으면 grad NaN. 호출자가 a > 0 을 보장해야 함.

  지수/밑이 상수인 경우는 `PowConstExp` / `PowConstBase` 로 dispatch (functional 참조).
  """

  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """거듭제곱 순전파 y = a ** b (원소별) 를 계산한다."""
    return a**b

  def backward_binary(
    self, grad: Array[D], a: Array[D], b: Array[D]
  ) -> tuple[Array[D], Array[D]]:
    """y = a ** b 의 체인룰 (원소별).

    ∂y/∂a = b * a ** (b - 1), ∂y/∂b = a ** b * log(a).
    """
    y = a**b
    dy_da = b * a ** (b - 1)
    dy_db = y * cast(Array[D], xp.log(cast(Array, a)))
    return (grad * dy_da, grad * dy_db)


class PowConstExp[D: DType](UnaryOp[D]):
  """지수가 상수인 거듭제곱: y = x ** n (n: Scalar).

  `Pow` (둘 다 Node) 를 쓰지 않고 별도 Op 인 이유:
  - `Pow.backward` 는 ∂y/∂b 항으로 `xp.log(a)` 를 계산하는데, n 이 상수이면 ∂y/∂n
    이 필요 없으니 log 항도 불필요. 즉 a 가 음수여도 NaN 발생 없음.
  - 상수는 `_inputs` 가 아니라 Op 인스턴스 필드로 보관 → `_inputs = (x,)` 로 깔끔.
  """

  def __init__(self, n: Scalar):
    self.n = n

  def forward_unary(self, x: Array[D]) -> Array[D]:
    """순전파 y = x ** n."""
    return x**self.n

  def backward_unary(self, grad: Array[D], x: Array[D]) -> Array[D]:
    """y = x ** n 의 체인룰 (n 은 상수). ∂y/∂x = n * x ** (n - 1)."""
    return grad * self.n * x ** (self.n - 1)


class PowConstBase[D: DType](UnaryOp[D]):
  """밑이 상수인 거듭제곱: y = c ** x (c: Scalar, c > 0).

  c 가 상수이므로 ∂y/∂c 는 필요 없고, ∂y/∂x = c^x * log(c) 만 계산.
  c > 0 이어야 log(c) 가 정의됨 — c <= 0 은 의미상 잘 쓰이지 않으니 호출자 책임으로 둠.
  """

  def __init__(self, c: Scalar):
    self.c = c
    self._log_c = math.log(c)

  def forward_unary(self, x: Array[D]) -> Array[D]:
    """순전파 y = c ** x."""
    return self.c**x

  def backward_unary(self, grad: Array[D], x: Array[D]) -> Array[D]:
    """y = c ** x 의 체인룰 (c 는 상수). ∂y/∂x = c ** x * log(c)."""
    y = self.c**x
    return grad * y * self._log_c
