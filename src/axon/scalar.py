"""Python primitive scalar 타입.

`functional` dispatch 와 `Op` 인스턴스 필드 (상수 지수/밑 등) 처럼 Node 와 함께
받거나 보관할 수 있는 상수값을 표현. backend Array 의 `__add__` / `__pow__` 등
연산자 시그니처가 `Self | int | float` 인 것과 결을 맞춘다.

`type Scalar = int | float` (PEP 695) 형태가 아닌 평범한 union 으로 둔 이유:
이렇게 두면 `isinstance(x, Scalar)` 가 그대로 동작한다 (UnionType 은 isinstance
지원). PEP 695 alias 는 `TypeAliasType` 인스턴스라 isinstance 에 직접 못 넣음.
"""

Scalar = int | float
