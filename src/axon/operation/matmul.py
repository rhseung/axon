from __future__ import annotations

from typing import TYPE_CHECKING

from axon.dtype import DType
from axon.errors import ShapeError
from axon.operation.op import BinaryOp

if TYPE_CHECKING:
  from axon.backend.protocol import Array


class MatMul[D: DType](BinaryOp[D]):
  def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]:
    """행렬 곱셈 순전파 y = a @ b 를 계산한다."""
    return a @ b

  def backward_binary(
    self,
    grad: Array[D],
    a: Array[D],
    b: Array[D],
    *,
    needs_grad: tuple[bool, bool],
  ) -> tuple[Array[D] | None, Array[D] | None]:
    """행렬 곱셈의 체인룰. ∂L/∂a = (∂L/∂y) @ b.T, ∂L/∂b = a.T @ (∂L/∂y)."""
    return (
      grad @ b.T if needs_grad[0] else None,
      a.T @ grad if needs_grad[1] else None,
    )

  def validate_shape(self, a: tuple[int, ...], b: tuple[int, ...]) -> None:
    """matmul shape 규칙 — BinaryOp 의 elementwise (a == b) 검증을 override.
    0-D scalar 케이스는 부모 `_validate` 가 미리 통과시키므로 여기선 multi-dim
    규칙만 강제.

    - 두 입력 모두 `ndim >= 2` 필수. 1-D 입력 시 numpy 는 `(n,) → (1,n)` /
      `(n,1)` 으로 자동 promote 하지만, axon 은 `(3,) ≠ (3,1)` 정책에 따라
      이 silent 변환을 금지 — 호출자가 `reshape` 로 명시.
    - 수축 차원: `a[-1] == b[-2]`.
    - 배치 차원 (`...` 앞쪽): broadcasting 금지 — 정확히 같아야 함.
    """
    if len(a) < 2 or len(b) < 2:
      raise ShapeError(
        f"MatMul: 두 입력 모두 ndim >= 2 필요 (1-D promotion 금지). "
        f"a.shape={a}, b.shape={b}"
      )

    if a[-1] != b[-2]:
      raise ShapeError(
        f"MatMul: 수축 차원 불일치. a.shape={a}, b.shape={b} "
        f"(a[-1]={a[-1]} != b[-2]={b[-2]})"
      )

    if a[:-2] != b[:-2]:
      raise ShapeError(
        f"MatMul: 배치 차원 불일치 — broadcasting 미지원. a.shape={a}, b.shape={b}"
      )
