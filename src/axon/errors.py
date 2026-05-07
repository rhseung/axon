from __future__ import annotations


class ShapeError(ValueError):
  """축 / shape 불일치. axon 은 broadcasting 을 허용하지 않으므로, backend 가
  silent 하게 broadcast 하기 전에 Op precheck 단에서 명시적으로 던진다.

  Op 별 규칙:
    - elementwise BinaryOp (`Add` / `Mul` / `Div` / `Pow`): `a.shape == b.shape`
    - `MatMul`: 수축 차원 일치 + 배치 차원 정확히 일치 (broadcast 금지)
    - `UnaryOp`: 기본 검증 없음. 서브클래스가 `validate_shape` override 가능.
  """

  pass
