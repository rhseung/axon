# axon — 백엔드 추상화 레이어

> NumPy / MLX / CuPy를 동일한 인터페이스로 추상화하는 설계 문서.

---

## 목표

- `axon.set_backend("mlx")` 한 줄로 백엔드 전환
- Op 구현체가 백엔드를 직접 import하지 않음
- MLX 특이사항(lazy eval, in-place 없음)을 추상화 레이어에서 처리
- NumPy를 기준 구현으로, MLX/CuPy는 프로토콜만 맞추면 교체 가능

---

## 파일 구조

```
axon/
├── dtype.py             # DType enum — 공개 API. Tensor 제네릭 bound.
└── backend/
    ├── __init__.py      # set_backend, get_backend, xp 프록시 노출
    ├── protocol.py      # Array / BackendProtocol / RandomProtocol 정의
    ├── _dtype.py        # axon DType ↔ 백엔드 native dtype 매핑
    ├── _numpy.py        # NumPy 구현 (기준)
    ├── _mlx.py          # MLX 구현 + MLXArray 래퍼
    └── _cupy.py         # CuPy 구현 (옵션)
```

`DType` 은 `axon/dtype.py` 에서 정의하고 `axon/__init__.py` 에서 재노출한다.
사용자가 `from axon import DType` 또는 `axon.DType.FLOAT32` 처럼 직접 쓸 수 있도록 공개 API다.
`Tensor` 의 제네릭도 `class Tensor[D: DType]` 로 axon `DType` 을 bound 한다.

---

## `ArrayLike` — Tensor와 Array의 공통 기반

`Tensor`와 `Array`는 연산자 이름이 같지만 반환 타입이 다르다.

```
Array.__add__(a, b) → Array   (즉시 계산, raw array)
Tensor.__add__(a, b)  → Tensor    (Add Op 생성, 그래프 노드)
```

반환 타입이 달라 `Tensor(Array)`로 상속하면 LSP 위반이다.
대신 공통 속성만 담은 `ArrayLike` Protocol을 기반으로 둔다.

```python
# axon/backend/protocol.py

class ArrayLike(Protocol):
    """Tensor와 Array의 공통 인터페이스 — 속성만 공유."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> DType: ...   # axon DType Enum
```

`Array`와 `Tensor` 모두 `ArrayLike`를 만족한다.
연산자는 각자 정의하되 반환 타입만 다르다.

```python
class Array(ArrayLike, Protocol):
    def __add__(self, other: Array | float) -> Array: ...
    # raw array 반환

class Tensor(ArrayLike):           # 실제 클래스, Protocol 아님
    def __add__(self, other: Tensor) -> Tensor: ...
    # Add Op 생성 후 Tensor 반환
```

---

## Protocol 정의

두 개의 Protocol이 필요하다.

- `Array` — array 객체 자체의 인터페이스 (연산자 오버로딩 포함)
- `BackendProtocol` — array를 생성/조작하는 백엔드 인스턴스의 인터페이스

```python
# axon/backend/protocol.py

from typing import Protocol, Any
from axon.dtype import DType


class Array(Protocol):
    """백엔드 array 객체의 공통 인터페이스.

    np.ndarray, mx.array, cp.ndarray 모두 이 프로토콜을 만족해야 한다.

    numpy.typing.NDArray와 이름 충돌을 피하기 위해 `Array`로 명명한다.
    """

    # --- 속성 ---
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> DType: ...   # axon DType (native dtype 아님 — 백엔드 구현이 매핑)

    # --- 산술 연산자 ---
    def __add__(self, other: "Array | float") -> "Array": ...
    def __radd__(self, other: "Array | float") -> "Array": ...
    def __sub__(self, other: "Array | float") -> "Array": ...
    def __rsub__(self, other: "Array | float") -> "Array": ...
    def __mul__(self, other: "Array | float") -> "Array": ...
    def __rmul__(self, other: "Array | float") -> "Array": ...
    def __truediv__(self, other: "Array | float") -> "Array": ...
    def __rtruediv__(self, other: "Array | float") -> "Array": ...
    def __pow__(self, other: "Array | float") -> "Array": ...
    def __neg__(self) -> "Array": ...
    def __abs__(self) -> "Array": ...
    def __matmul__(self, other: "Array") -> "Array": ...

    # --- 비교 연산자 ---
    def __eq__(self, other: object) -> "Array": ...  # type: ignore[override]
    def __lt__(self, other: "Array | float") -> "Array": ...
    def __le__(self, other: "Array | float") -> "Array": ...
    def __gt__(self, other: "Array | float") -> "Array": ...
    def __ge__(self, other: "Array | float") -> "Array": ...

    # --- 인덱싱 ---
    def __getitem__(self, idx: Any) -> "Array": ...
    def __setitem__(self, idx: Any, value: Any) -> None: ...

    # --- 속성/변환 ---
    @property
    def T(self) -> "Array": ...

    def tolist(self) -> list: ...
    def astype(self, dtype: DType) -> "Array": ...


class BackendProtocol(Protocol):
    """array를 생성/조작하는 백엔드 인스턴스의 인터페이스.

    Array가 직접 지원하지 않는 연산(축소, 형상 변환 등)을 담당한다.
    """

    # --- 생성 ---
    def array(self, data: Any, dtype: DType = ...) -> Array: ...
    def zeros(self, shape: tuple, dtype: DType = ...) -> Array: ...
    def ones(self, shape: tuple, dtype: DType = ...) -> Array: ...
    def full(self, shape: tuple, fill_value: float, dtype: DType = ...) -> Array: ...
    def full_like(self, x: Array, fill_value: float) -> Array: ...
    def zeros_like(self, x: Array) -> Array: ...
    def ones_like(self, x: Array) -> Array: ...
    def eye(self, n: int, dtype: DType = ...) -> Array: ...
    def arange(self, *args, **kwargs) -> Array: ...

    # --- 수학 함수 ---
    def exp(self, x: Array) -> Array: ...
    def log(self, x: Array) -> Array: ...
    def sqrt(self, x: Array) -> Array: ...
    def rsqrt(self, x: Array) -> Array: ...       # 1/sqrt(x) — LayerNorm 최적화
    def abs(self, x: Array) -> Array: ...
    def sign(self, x: Array) -> Array: ...        # Lion optimizer
    def maximum(self, a: Array, b: Array) -> Array: ...
    def minimum(self, a: Array, b: Array) -> Array: ...
    def clip(self, x: Array, a_min: float, a_max: float) -> Array: ...
    def sin(self, x: Array) -> Array: ...
    def cos(self, x: Array) -> Array: ...
    def tanh(self, x: Array) -> Array: ...        # GELU 등 활성화

    # --- 축소 ---
    def sum(self, x: Array, axis: int | tuple | None = None, keepdims: bool = False) -> Array: ...
    def mean(self, x: Array, axis: int | tuple | None = None, keepdims: bool = False) -> Array: ...
    def var(self, x: Array, axis: int | tuple | None = None, keepdims: bool = False, ddof: int = 0) -> Array: ...
    def max(self, x: Array, axis: int | tuple | None = None, keepdims: bool = False) -> Array: ...
    def min(self, x: Array, axis: int | tuple | None = None, keepdims: bool = False) -> Array: ...
    def norm(self, x: Array, ord: int | float | None = None, axis: int | tuple | None = None, keepdims: bool = False) -> Array: ...   # gradient clipping

    # --- 형상 ---
    def reshape(self, x: Array, shape: tuple) -> Array: ...
    def transpose(self, x: Array, axes: tuple | None = None) -> Array: ...
    def expand_dims(self, x: Array, axis: int | tuple) -> Array: ...   # unsqueeze
    def squeeze(self, x: Array, axis: int | tuple | None = None) -> Array: ...
    def flatten(self, x: Array, start_dim: int = 0) -> Array: ...
    def broadcast_to(self, x: Array, shape: tuple) -> Array: ...
    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array: ...
    def stack(self, arrays: list[Array], axis: int = 0) -> Array: ...
    def split(self, x: Array, indices_or_sections: int | list[int], axis: int = 0) -> list[Array]: ...
    def flip(self, x: Array, axis: int | tuple | None = None) -> Array: ...   # RoPE 등

    # --- 선형대수 ---
    def matmul(self, a: Array, b: Array) -> Array: ...
    def einsum(self, subscripts: str, *operands: Array) -> Array: ...  # attention 계산

    # --- 인덱싱 ---
    def where(self, condition: Array, x: Array, y: Array) -> Array: ...
    def take(self, x: Array, indices: Array, axis: int | None = None) -> Array: ...
    def tril(self, x: Array, k: int = 0) -> Array: ...    # causal mask
    def triu(self, x: Array, k: int = 0) -> Array: ...

    # --- 정렬 ---
    def sort(self, x: Array, axis: int = -1) -> Array: ...
    def argsort(self, x: Array, axis: int = -1) -> Array: ...   # top-k sampling

    # --- 패딩 ---
    def pad(self, x: Array, pad_width: tuple, constant_values: float = 0) -> Array: ...   # Conv

    # --- 변환 ---
    def to_numpy(self, x: Array) -> "np.ndarray": ...
    def from_numpy(self, x: "np.ndarray") -> Array: ...

    # --- 난수 ---
    @property
    def random(self) -> "RandomProtocol": ...   # NumPy 의 np.random 패턴 — 백엔드 인스턴스가 보유


class RandomProtocol(Protocol):
    """난수 생성 — 가중치 초기화와 Dropout에 필요. `BackendProtocol.random` 으로 접근."""

    def normal(self, shape: tuple, mean: float = 0.0, std: float = 1.0, dtype: DType = ...) -> Array: ...
    def uniform(self, shape: tuple, low: float = 0.0, high: float = 1.0, dtype: DType = ...) -> Array: ...
    def bernoulli(self, p: float, shape: tuple, dtype: DType = ...) -> Array: ...   # Dropout 마스크
    def seed(self, s: int) -> None: ...
```

각 백엔드는 `BackendProtocol`을 구현하고, `random` 속성으로 `RandomProtocol` 인스턴스를 보유한다.
`xp` 로 일반 연산, `xp.random` 으로 난수 생성에 접근한다 (NumPy 의 `np.random` 패턴).

```python
from axon.backend import xp

# 일반 연산
xp.tril(xp.ones((T, T)))             # causal mask
xp.var(x, axis=-1, keepdims=True)    # LayerNorm

# 난수
xp.random.normal((out, in_), std=0.02)   # GPT-2 초기화
xp.random.bernoulli(0.1, x.shape)        # Dropout 마스크
```

### 역할 분리

`Array`는 array 객체가 직접 지원하는 연산 — 연산자 오버로딩, 속성 접근, 인덱싱.
`BackendProtocol`은 array 객체만으로 표현하기 어려운 연산 — 생성, 축소, 형상 변환, grad 관리.

Op 구현에서 단순 산술은 Array 연산자를 직접 쓰고, 나머지는 `xp`를 통한다.

```python
class Add(BinaryOp):
    def forward(self, a, b):
        return Tensor(a._data + b._data)   # Array.__add__ 직접 사용

class Exp(UnaryOp):
    def forward(self, x):
        return Tensor(xp.exp(x._data))   # BackendProtocol 경유
```

---

## `__init__.py` — 백엔드 전환 및 `xp` 프록시

### `xp` 컨벤션

NumPy/CuPy 커뮤니티에서 array library를 통칭할 때 `xp` 를 쓰는 게 관례다.
axon 도 이 컨벤션을 따른다. 다만 **모듈 `__getattr__` 위임은 쓰지 않는다** —
정적 타입 분석이 `Any` 로 죽기 때문에 pyrefly/IDE 자동완성이 동작하지 않는다.

대신 `BackendProtocol` 을 흉내내는 **프록시 객체** 를 만들어 `xp` 라는 이름으로 노출한다.
프록시는 `__getattribute__` 로 매번 현재 백엔드에 위임하므로 `set_backend()` 이후에도 자동 반영된다.
정적 타입에선 `xp: BackendProtocol` 로 보이므로 `xp.exp`, `xp.random.normal` 등이 모두 타입 추론된다.

```python
# axon/backend/__init__.py

from __future__ import annotations
from typing import Literal, cast
from axon.backend.protocol import BackendProtocol

_current: BackendProtocol | None = None
_name: str = "numpy"


def set_backend(name: Literal["numpy", "mlx", "cupy"] = "numpy"):
    global _current, _name

    match name:
        case "numpy":
            from axon.backend._numpy import NumpyBackend
            _current = NumpyBackend()
        case "mlx":
            from axon.backend._mlx import MLXBackend
            _current = MLXBackend()
        case "cupy":
            from axon.backend._cupy import CuPyBackend
            _current = CuPyBackend()
        case _:
            raise ValueError(f"알 수 없는 백엔드: {name!r}")

    _name = name


def get_backend() -> str:
    return _name


def current() -> BackendProtocol:
    if _current is None:
        set_backend("numpy")
    return _current  # type: ignore


class _BackendProxy:
    """현재 백엔드 인스턴스에 모든 속성 접근을 위임한다.

    `xp.exp(x)` → `current().exp(x)` 로 자동 위임.
    `set_backend()` 이후에도 같은 `xp` 변수가 새 백엔드를 가리킨다.
    """

    __slots__ = ()

    def __getattribute__(self, name: str):
        return getattr(current(), name)


# 정적 타입은 BackendProtocol 로 보이지만 런타임은 프록시.
xp: BackendProtocol = cast(BackendProtocol, _BackendProxy())


# 기본 백엔드 초기화
set_backend("numpy")
```

Op 은 프록시를 import 해서 쓴다.

```python
# Op 구현 예시
from axon.backend import xp

class Exp(UnaryOp):
    def forward(self, x):
        return Tensor(xp.exp(x._data))

    def backward(self, grad, x):
        return (Tensor(grad._data * xp.exp(x._data)),)

class Sum(UnaryOp):
    def forward(self, x):
        return Tensor(xp.sum(x._data, axis=self.axis, keepdims=self.keepdims))
```

`current().exp(x)` 대신 `xp.exp(x)` — 짧고, NumPy 코드와 동일한 패턴이며,
`xp: BackendProtocol` 정적 타입 덕에 `pyrefly` / IDE 가 모든 멤버를 추론한다.

---

## `_numpy.py` — 기준 구현

```python
# axon/backend/_numpy.py

import numpy as np
from numpy.typing import DTypeLike
from axon.backend.protocol import Array


class NumpyBackend:
    # --- 생성 ---
    def array(self, data, dtype=np.float32):
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float32):
        return np.ones(shape, dtype=dtype)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def ones_like(self, x):
        return np.ones_like(x)

    def arange(self, *args, **kwargs):
        return np.arange(*args, **kwargs)

    # --- 산술 ---
    def add(self, a, b): return np.add(a, b)
    def multiply(self, a, b): return np.multiply(a, b)
    def matmul(self, a, b): return np.matmul(a, b)
    def power(self, x, n): return np.power(x, n)
    def negative(self, x): return np.negative(x)

    # --- 수학 함수 ---
    def exp(self, x): return np.exp(x)
    def log(self, x): return np.log(x)
    def sqrt(self, x): return np.sqrt(x)
    def abs(self, x): return np.abs(x)
    def maximum(self, a, b): return np.maximum(a, b)
    def minimum(self, a, b): return np.minimum(a, b)
    def clip(self, x, a_min, a_max): return np.clip(x, a_min, a_max)
    def sin(self, x): return np.sin(x)
    def cos(self, x): return np.cos(x)

    # --- 축소 ---
    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return np.min(x, axis=axis, keepdims=keepdims)

    # --- 형상 ---
    def reshape(self, x, shape): return np.reshape(x, shape)
    def transpose(self, x, axes=None): return np.transpose(x, axes)
    def expand_dims(self, x, axis): return np.expand_dims(x, axis)
    def squeeze(self, x, axis=None): return np.squeeze(x, axis)
    def concatenate(self, arrays, axis=0): return np.concatenate(arrays, axis)
    def stack(self, arrays, axis=0): return np.stack(arrays, axis)

    # --- 인덱싱 ---
    def where(self, condition, x, y): return np.where(condition, x, y)
    def take(self, x, indices, axis=None): return np.take(x, indices, axis)

    # --- 변환 ---
    def to_numpy(self, x): return np.asarray(x)
    def from_numpy(self, x): return np.asarray(x)

    # --- 기타 ---
    def eval(self, *arrays): pass  # NumPy는 eager — no-op
```

---

## `_mlx.py` — MLX 특이사항 처리

MLX와 NumPy의 주요 차이점:


| 항목          | NumPy         | MLX                |
| ----------- | ------------- | ------------------ |
| 실행 방식       | Eager         | Lazy (명시적 eval 필요) |
| In-place 연산 | `+=`, `-=` 가능 | 불가, 재할당 필요         |
| dtype       | `np.float32`  | `mx.float32`       |
| 컴파일         | 없음            | `mx.compile` 가능    |


```python
# axon/backend/_mlx.py

import mlx.core as mx
from axon.backend.protocol import Array


class MLXBackend:
    # --- 생성 ---
    def array(self, data, dtype=mx.float32):
        return mx.array(data, dtype=dtype)

    def zeros(self, shape, dtype=mx.float32):
        return mx.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=mx.float32):
        return mx.ones(shape, dtype=dtype)

    def zeros_like(self, x):
        return mx.zeros_like(x)

    def ones_like(self, x):
        return mx.ones_like(x)

    def arange(self, *args, **kwargs):
        return mx.arange(*args, **kwargs)

    # --- 산술 ---
    def add(self, a, b): return mx.add(a, b)
    def multiply(self, a, b): return mx.multiply(a, b)
    def matmul(self, a, b): return mx.matmul(a, b)
    def power(self, x, n): return mx.power(x, n)
    def negative(self, x): return mx.negative(x)

    # --- 수학 함수 ---
    def exp(self, x): return mx.exp(x)
    def log(self, x): return mx.log(x)
    def sqrt(self, x): return mx.sqrt(x)
    def abs(self, x): return mx.abs(x)
    def maximum(self, a, b): return mx.maximum(a, b)
    def minimum(self, a, b): return mx.minimum(a, b)
    def clip(self, x, a_min, a_max): return mx.clip(x, a_min, a_max)
    def sin(self, x): return mx.sin(x)
    def cos(self, x): return mx.cos(x)

    # --- 축소 ---
    def sum(self, x, axis=None, keepdims=False):
        return mx.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return mx.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x, axis=None, keepdims=False):
        return mx.max(x, axis=axis, keepdims=keepdims)

    def min(self, x, axis=None, keepdims=False):
        return mx.min(x, axis=axis, keepdims=keepdims)

    # --- 형상 ---
    def reshape(self, x, shape): return mx.reshape(x, shape)
    def transpose(self, x, axes=None): return mx.transpose(x, axes)
    def expand_dims(self, x, axis): return mx.expand_dims(x, axis)
    def squeeze(self, x, axis=None): return mx.squeeze(x, axis)
    def concatenate(self, arrays, axis=0): return mx.concatenate(arrays, axis)
    def stack(self, arrays, axis=0): return mx.stack(arrays, axis)

    # --- 인덱싱 ---
    def where(self, condition, x, y): return mx.where(condition, x, y)
    def take(self, x, indices, axis=None): return mx.take(x, indices, axis)

    # --- 변환 ---
    def to_numpy(self, x):
        import numpy as np
        mx.eval(x)            # lazy → 실제 값 강제
        return np.array(x)    # mx.array는 __array__ 지원
    def from_numpy(self, x):
        import numpy as np
        return mx.array(np.asarray(x))

    # --- Lazy eval ---
    def eval(self, *arrays):
        mx.eval(*arrays)
```

### In-place / `__setitem__` / lazy eval

**MLX 0.31+ 에서는 별도 래퍼가 필요없다.** `mx.array` 가 다음을 모두 native 로 지원한다:

- `arr[idx] = value` (실제로 in-place — `id()` 가 동일하게 유지됨)
- `arr += other`, `-=`, `*=` (in-place)
- `np.array(mx_array)` (자동 변환)

따라서 `MLXBackend` 는 `mx.array` 를 그대로 반환하고, `Tensor._data` 도 `mx.array` 를 직접 보유한다.
`backward()` 의 grad 누적(`grads[id(inp)] += ig._data`) 도 NumPy 와 완전히 동일하게 동작한다.

```python
class MLXBackend:
    def zeros_like(self, x): return mx.zeros_like(x)
    def exp(self, x): return mx.exp(x)
    def to_numpy(self, x):
        mx.eval(x)         # lazy → 강제 평가
        return np.array(x)
```

#### MLX 가 NumPy 와 다른 부분 (래퍼 없이 직접 우회)

| 항목 | 처리 |
|---|---|
| Lazy eval | `to_numpy()` / `eval()` 에서만 `mx.eval()` 강제 |
| `mx.flip` 부재 | 슬라이싱 (`x[::-1]`) 으로 직접 구현 |
| `bfloat16` ↔ NumPy 호환 | NumPy 에는 native bfloat16 없음 — `_dtype.py` 매핑에서 fp32 로 대체 |

#### 만약 MLX 가 immutable 로 회귀하면

과거 MLX 버전에서는 `__setitem__` / `__iadd__` 가 없었다.
그 경우 `MLXArray` 래퍼로 재할당 시뮬레이션이 필요하지만,
현재 버전 (0.31+) 에서는 불필요하므로 추가하지 않는다.

### Lazy eval 시점

`mx.eval()` 은 `to_numpy()` / `eval()` 메서드에서 명시적으로 호출한다.
사용자 코드에서 값을 실제로 읽는 시점 — `Tensor.as_numpy()`, `Tensor.item()`, 로깅 등 — 에 한해 발생한다.
`backward()` 안에서 명시적으로 eval 을 호출할 필요는 없다 (다음 forward 까지 lazy 그래프가 누적되며,
다음 사용 시점에 한 번에 eval).

---

## DType — 공개 Enum + 백엔드 매핑

문자열이나 `np.float32` 같은 백엔드 종속 타입 대신 axon 전용 `DType` Enum을 사용한다.
**`DType` 은 공개 API** 다 — `from axon import DType` 으로 직접 import 가능하고,
`Tensor` 의 제네릭도 `class Tensor[D: DType]` 로 이걸 bound 한다.
`Tensor.dtype` 은 axon `DType` 을 반환한다.
한편 `Array.dtype` 은 백엔드 native (`np.dtype` / `mx.Dtype` / `cp.dtype`) 그대로 노출된다 —
np/mx/cp 를 직접 wrap 하지 않기 위함이고, 사용자 surface 에는 노출되지 않으므로 무방하다.
Tensor 가 `from_backend_dtype()` 으로 native → axon DType 변환을 담당한다.

파일은 두 곳으로 분리한다:

- `axon/dtype.py` — `DType` enum 자체. 다른 axon 코드가 자유롭게 import.
- `axon/backend/_dtype.py` — `DType` ↔ 백엔드 native 타입 매핑 + `to_backend_dtype()`.

### 지원 dtype

세 백엔드의 교집합 + 백엔드별 예외로 구성한다 (MLX 0.31 / Apple Metal 기준 실측).

| axon DType | NumPy | MLX | CuPy | 비고 |
| --- | --- | --- | --- | --- |
| FLOAT16 | ✓ | ✓ | ✓ | |
| BFLOAT16 | △ (fp32 로 대체) | ✓ | ✗ | NumPy 는 native bfloat16 없음 — fp32 매핑. CuPy 미지원. |
| FLOAT32 | ✓ | ✓ | ✓ | 기본값 |
| FLOAT64 | ✓ | ✗ | ✓ | MLX 는 GPU 미지원 (CPU stream 만 가능) — 차단. |
| INT32 | ✓ | ✓ | ✓ | |
| INT64 | ✓ | ✓ | ✓ | |
| BOOL | ✓ | ✓ | ✓ | |

MLX 에서 `FLOAT64`, CuPy 에서 `BFLOAT16` 을 사용하면 즉시 `TypeError`.
폴백 없음 — 잘못된 dtype 은 조용히 넘어가지 않는다.
fp64 가 필요한 경우 (예: numerical gradient check) 는 NumPy 백엔드를 쓴다.

> **CuPy 와 bfloat16** — CuPy 는 `cupy.bfloat16` top-level export 가 없다 (numpy 1.x 가 native bfloat16 부재인 데 따른 것). 부분 지원이 `ml_dtypes` 패키지 + CUDA ≥ 12.2 + NumPy ≥ 2.1.2 조합에서만 가능하므로, 디폴트 정책으로는 차단한다. 향후 옵션화 여지 있음.

```python
# axon/dtype.py — 공개 API

from enum import Enum, auto


class DType(Enum):
    FLOAT16  = auto()
    BFLOAT16 = auto()
    FLOAT32  = auto()
    FLOAT64  = auto()
    INT32    = auto()
    INT64    = auto()
    BOOL     = auto()
```

```python
# axon/backend/_dtype.py — 백엔드 native 타입과의 매핑

from typing import Any
from axon.dtype import DType
from axon.backend import get_backend


# 백엔드별 미지원 dtype — 사용 시 즉시 TypeError
_UNSUPPORTED: dict[str, set[DType]] = {
    "numpy": set(),
    "mlx":   {DType.FLOAT64},   # GPU 미지원 (CPU stream 만 동작)
    "cupy":  {DType.BFLOAT16},
}


def _build_numpy_map():
    import numpy as np
    return {
        DType.FLOAT16:  np.float16,
        DType.BFLOAT16: np.float32,  # numpy 1.24+ 지원, 그 이하는 float32
        DType.FLOAT32:  np.float32,
        DType.FLOAT64:  np.float64,
        DType.INT32:    np.int32,
        DType.INT64:    np.int64,
        DType.BOOL:     np.bool_,
    }


def _build_mlx_map():
    import mlx.core as mx
    return {
        DType.FLOAT16:  mx.float16,
        DType.BFLOAT16: mx.bfloat16,
        DType.FLOAT32:  mx.float32,
        DType.INT32:    mx.int32,
        DType.INT64:    mx.int64,
        DType.BOOL:     mx.bool_,
        # FLOAT64 없음 — _UNSUPPORTED 에서 차단 (GPU 미지원)
    }


def _build_cupy_map():
    import cupy as cp
    return {
        DType.FLOAT16:  cp.float16,
        DType.FLOAT32:  cp.float32,
        DType.FLOAT64:  cp.float64,
        DType.INT32:    cp.int32,
        DType.INT64:    cp.int64,
        DType.BOOL:     cp.bool_,
        # BFLOAT16 없음 — to_backend_dtype 에서 에러
    }


_DTYPE_BUILDERS = {
    "numpy": _build_numpy_map,
    "mlx":   _build_mlx_map,
    "cupy":  _build_cupy_map,
}


def to_backend_dtype(dtype: DType) -> Any:
    """axon DType → 현재 백엔드의 실제 dtype 으로 변환.

    미지원 dtype 은 폴백 없이 즉시 TypeError.
    """
    name = get_backend()
    if dtype in _UNSUPPORTED[name]:
        raise TypeError(
            f"{dtype.name} is not supported on the {name!r} backend. "
            f"Switch to a supported backend first: axon.set_backend('numpy')"
        )
    return _DTYPE_BUILDERS[name]()[dtype]


def from_backend_dtype(native: Any) -> DType:
    """백엔드 native dtype → axon DType. Array.dtype 구현에서 사용."""
    name = get_backend()
    for dt, n in _DTYPE_BUILDERS[name]().items():
        if n == native:
            return dt
    raise TypeError(f"Unknown native dtype: {native!r} on backend {name!r}")
```

`axon/__init__.py` 에서 재노출하면 사용자가 `axon.DType.FLOAT32` 로 직접 쓸 수 있다.

```python
# axon/__init__.py
from axon.dtype import DType
```

numerical gradient check처럼 float64가 필요한 경우 NumPy나 CuPy 백엔드를 명시적으로 사용한다.

```python
def check_gradients(model, x, *, backend: str = "numpy"):
    """수치 grad 검증은 float64가 필요 — 기본값 numpy."""
    original = get_backend()
    set_backend(backend)
    try:
        ...
    finally:
        set_backend(original)
```

---

## 테스트 전략

### 백엔드 동일 결과 검증

```python
# tests/test_backends.py

import pytest
import numpy as np

def test_matmul_parity():
    """numpy ↔ mlx 결과 일치 검증"""
    x = np.random.randn(4, 3).astype(np.float32)
    w = np.random.randn(3, 5).astype(np.float32)

    axon.set_backend("numpy")
    out_np = (Tensor(x) @ Tensor(w)).as_numpy()

    axon.set_backend("mlx")
    out_mlx = (Tensor(x) @ Tensor(w)).as_numpy()

    np.testing.assert_allclose(out_np, out_mlx, atol=1e-5)
```

모든 Op에 대해 `numpy ↔ mlx` 동일 결과를 검증한다.

### float64 미지원 예외 처리

```python
# numerical gradient check는 float64가 필요
# mlx 백엔드에서는 자동으로 numpy로 fallback

def check_gradients(model, x, backend_override="numpy"):
    original = get_backend()
    set_backend(backend_override)
    # ... gradient check
    set_backend(original)
```

---

## 의존 순서

지금 백엔드 추상화를 끝까지 박고 — 프론트(`Tensor`/`Op`/`functional`) 도 한 번에 갈아끼운다.
세 백엔드 모두 구현하되, **로컬 개발 루프(macOS)는 NumPy ↔ MLX 로 회전**하고
CuPy 는 외장 GPU / Colab / Windows 등 NVIDIA 환경에서 별도로 검증한다.

```text
1. axon/dtype.py — DType enum (공개)
2. axon/backend/protocol.py — Array / BackendProtocol / RandomProtocol
3. axon/backend/_dtype.py — to_backend_dtype / from_backend_dtype
4. axon/backend/_numpy.py — NumpyBackend (기준 구현)
5. axon/backend/_mlx.py — MLXArray 래퍼 + MLXBackend
6. axon/backend/_cupy.py — CuPyBackend
7. axon/backend/__init__.py — set_backend, xp 프록시
8. 프론트 마이그레이션:
   - Tensor[D: DType] / Tensor._data: Array 로 변경
   - 기존 Op 들 (Add/Mul/Pow/MatMul) 의 np 직접 참조 → xp 로 교체
   - functional / parameter / backward 도 동일하게 교체
9. 테스트:
   - 로컬 (macOS): NumPy ↔ MLX parity, DType 매핑, MLXArray in-place 누적
   - 원격 (Colab / Windows): NumPy ↔ CuPy parity (pytest 마커로 분리)
10. @axon.jit (mx.compile / cuda graph 위임) — 선택
```

Op 구현체는 `xp.xxx` 만 호출하므로 백엔드 교체 시 수정 불필요한 것이 검증의 기준이다.

CUDA 테스트는 `pytest -m cuda` 로 분리해 두면 macOS 로컬 CI 에선 자동 skip 되고
NVIDIA 환경에선 따로 돌릴 수 있다.
