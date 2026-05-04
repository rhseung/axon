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
axon/backend/
├── __init__.py      # set_backend, get_backend, 현재 백엔드 노출
├── protocol.py      # BackendProtocol 정의
├── _numpy.py        # NumPy 구현 (기준)
├── _mlx.py          # MLX 구현 + 특이사항 처리
└── _cupy.py         # CuPy 구현 (옵션)
```

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
from numpy.typing import DTypeLike


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
    def dtype(self) -> Any: ...

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
    def astype(self, dtype: DTypeLike) -> "Array": ...


class BackendProtocol(Protocol):
    """array를 생성/조작하는 백엔드 인스턴스의 인터페이스.

    Array가 직접 지원하지 않는 연산(축소, 형상 변환 등)을 담당한다.
    """

    # --- 생성 ---
    def array(self, data: Any, dtype: DTypeLike = ...) -> Array: ...
    def zeros(self, shape: tuple, dtype: DTypeLike = ...) -> Array: ...
    def ones(self, shape: tuple, dtype: DTypeLike = ...) -> Array: ...
    def full(self, shape: tuple, fill_value: float, dtype: DTypeLike = ...) -> Array: ...
    def full_like(self, x: Array, fill_value: float) -> Array: ...
    def zeros_like(self, x: Array) -> Array: ...
    def ones_like(self, x: Array) -> Array: ...
    def eye(self, n: int, dtype: DTypeLike = ...) -> Array: ...
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


class RandomProtocol(Protocol):
    """난수 생성 — 가중치 초기화와 Dropout에 필요."""

    def normal(self, shape: tuple, mean: float = 0.0, std: float = 1.0, dtype: DTypeLike = ...) -> Array: ...
    def uniform(self, shape: tuple, low: float = 0.0, high: float = 1.0, dtype: DTypeLike = ...) -> Array: ...
    def bernoulli(self, p: float, shape: tuple, dtype: DTypeLike = ...) -> Array: ...   # Dropout 마스크
    def seed(self, s: int) -> None: ...
```

각 백엔드는 `BackendProtocol`과 `RandomProtocol`을 모두 구현한다.
`xp`로 일반 연산, `xp.random`으로 난수 생성에 접근한다.

```python
import axon.backend as xp

# 일반 연산
xp.tril(xp.ones((T, T)))             # causal mask
xp.var(x, axis=-1, keepdims=True)    # LayerNorm

# 난수
xp.random.normal((out, in_), std=0.02)   # GPT-2 초기화
xp.random.bernoulli(0.1, x.shape)        # Dropout 마스크
```

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

## `__init__.py` — 백엔드 전환 및 `xp` alias

### `xp` 컨벤션

NumPy/CuPy 커뮤니티에서 array library를 통칭할 때 `xp`를 쓰는 게 관례야.
axon도 이 컨벤션을 따른다. `__getattr__`로 위임해 `set_backend()` 이후에도 자동 반영된다.

```python
# axon/backend/__init__.py

from __future__ import annotations
from typing import Literal
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


def __getattr__(name: str):
    """import axon.backend as xp 후 xp.exp(x) 호출 시 current().exp(x)로 위임.

    set_backend() 이후 변경도 자동 반영된다.
    """
    return getattr(current(), name)


# 기본 백엔드 초기화
set_backend("numpy")
```

Op은 `import axon.backend as xp` 하나만 import한다.

```python
# Op 구현 예시
import axon.backend as xp

class Exp(UnaryOp):
    def forward(self, x):
        return Tensor(xp.exp(x._data))

    def backward(self, grad, x):
        return (Tensor(grad._data * xp.exp(x._data)),)

class Sum(UnaryOp):
    def forward(self, x):
        return Tensor(xp.sum(x._data, axis=self.axis, keepdims=self.keepdims))
```

`xp.exp(x)` 대신 `xp.exp(x)` — 훨씬 자연스럽고 NumPy 코드와 동일한 패턴이야.

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
    def to_numpy(self, x): return mx.eval(x) or x.tolist()  # mx array → numpy
    def from_numpy(self, x):
        import numpy as np
        return mx.array(np.asarray(x))

    # --- Lazy eval ---
    def eval(self, *arrays):
        mx.eval(*arrays)
```

### In-place 문제 해결 — MLXArray 래퍼

MLX array를 그대로 `Tensor._data`로 쓰면 `+=`가 안 된다.
`**MLXArray` 래퍼**를 만들어 `__iadd__`가 내부에서 재할당을 처리하게 한다.
`backward()`, `Parameter`는 NumPy와 완전히 동일하게 쓸 수 있다.

```python
# axon/backend/_mlx.py

import mlx.core as mx
import numpy as np


class MLXArray:
    """mx.array를 감싸 in-place 연산과 lazy eval을 투명하게 처리.

    - __iadd__, __isub__ 등이 내부 재할당으로 in-place를 시뮬레이션한다.
    - as_numpy(), tolist() 등 값을 실제로 읽을 때 mx.eval()을 호출한다.
    - 호출자는 NumPy ndarray와 동일하게 쓸 수 있다.
    """

    __slots__ = ("_mx",)

    def __init__(self, data: mx.array):
        self._mx = data

    # --- 속성 ---
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._mx.shape)

    @property
    def ndim(self) -> int:
        return self._mx.ndim

    @property
    def dtype(self):
        return self._mx.dtype

    @property
    def T(self) -> "MLXArray":
        return MLXArray(self._mx.T)

    # --- In-place 연산 (재할당을 내부에서 처리) ---
    def __iadd__(self, other: "MLXArray | float") -> "MLXArray":
        rhs = other._mx if isinstance(other, MLXArray) else other
        self._mx = self._mx + rhs   # 재할당은 여기서
        return self                  # self를 반환 → 호출자 재할당 불필요

    def __isub__(self, other: "MLXArray | float") -> "MLXArray":
        rhs = other._mx if isinstance(other, MLXArray) else other
        self._mx = self._mx - rhs
        return self

    def __imul__(self, other: "MLXArray | float") -> "MLXArray":
        rhs = other._mx if isinstance(other, MLXArray) else other
        self._mx = self._mx * rhs
        return self

    # --- 일반 연산 ---
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def __matmul__(self, other): ...
    def __neg__(self): ...
    # (나머지 연산자 동일 패턴)

    # --- 인덱싱 ---
    def __getitem__(self, idx):
        return MLXArray(self._mx[idx])

    def __setitem__(self, idx, value):
        # MLX는 setitem도 없음 — at[].set() 패턴으로 처리
        rhs = value._mx if isinstance(value, MLXArray) else value
        self._mx = self._mx.at[idx].set(rhs)

    # --- 변환 (값을 실제로 읽을 때 eval) ---
    def tolist(self) -> list:
        mx.eval(self._mx)
        return self._mx.tolist()

    def astype(self, dtype) -> "MLXArray":
        return MLXArray(self._mx.astype(dtype))

    def __array__(self, dtype=None):
        """np.asarray(mlx_array) 지원 — eval 자동 호출."""
        mx.eval(self._mx)
        arr = np.array(self._mx.tolist())
        return arr.astype(dtype) if dtype else arr
```

이제 `MLXBackend`는 `MLXArray`를 반환한다.

```python
class MLXBackend:
    def zeros_like(self, x: MLXArray) -> MLXArray:
        return MLXArray(mx.zeros_like(x._mx))

    def exp(self, x: MLXArray) -> MLXArray:
        return MLXArray(mx.exp(x._mx))

    def accumulate(self, target, delta):
        target += delta   # MLXArray.__iadd__ 호출 — 내부에서 재할당
        return target

    def eval_grads(self, *arrays):
        pass   # MLXArray는 읽을 때 자동 eval — 별도 호출 불필요
```

`**backward()`는 NumPy와 완전히 동일하다.**

```python
# axon/backward.py — 백엔드 분기, accumulate_grad, eval_grads 전혀 없음
def backward(loss: Tensor):
    grads: dict[int, Array] = {}
    grads[id(loss)] = xp.ones_like(loss._data)

    for node in reversed(topological_order(loss)):
        g = grads.get(id(node))
        if g is None:
            continue

        input_grads = node._op.backward(Tensor(g), *node._inputs)

        for inp, ig in zip(node._inputs, input_grads):
            if id(inp) not in grads:
                grads[id(inp)] = xp.zeros_like(inp._data)
            grads[id(inp)] += ig._data          # ← NumPy와 동일

            if isinstance(inp, Parameter):
                inp.grad._data += ig._data      # ← NumPy와 동일
```

`**Parameter`도 특별한 처리가 전혀 없다.**

```python
class Parameter(Tensor):
    def zero_grad(self):
        self.grad = Tensor(xp.zeros_like(self._data))

    # accumulate_grad 같은 메서드 불필요
```

### Lazy eval 시점

`mx.eval()`은 `MLXArray.__array__()` 또는 `tolist()`에서 자동 호출된다.
`as_numpy()`, `loss.item()`, 로깅 등 실제로 값을 읽는 시점에만 발생한다.
`backward()` 안에서 명시적으로 eval을 호출할 필요가 없다.

---

## dtype — Enum + 백엔드 매핑

문자열이나 `np.float32` 같은 백엔드 종속 타입 대신 axon 전용 `DType` Enum을 사용한다.
백엔드별 실제 dtype은 매핑 테이블에서 관리한다.

### 지원 dtype

세 백엔드의 교집합 + MLX 제외 예외 항목으로 구성한다.


| axon DType | NumPy | MLX | CuPy | 비고                      |
| ---------- | ----- | --- | ---- | ----------------------- |
| FLOAT16    | ✓     | ✓   | ✓    |                         |
| BFLOAT16   | ✓     | ✓   | ✗    | CuPy 미지원 — MLX 전용       |
| FLOAT32    | ✓     | ✓   | ✓    | 기본값                     |
| FLOAT64    | ✓     | ✗   | ✓    | MLX 미지원 — NumPy/CuPy 전용 |
| INT32      | ✓     | ✓   | ✓    |                         |
| INT64      | ✓     | ✗   | ✓    | MLX 미지원 — NumPy/CuPy 전용 |
| BOOL       | ✓     | ✓   | ✓    |                         |


MLX에서 `FLOAT64`, `INT64`, `BFLOAT16`(CuPy 제외) 을 사용하면 즉시 에러.
폴백 없음 — 잘못된 dtype은 조용히 넘어가지 않는다.

```python
# axon/backend/_dtype.py

from enum import Enum, auto
from typing import Any


class DType(Enum):
    FLOAT16  = auto()
    BFLOAT16 = auto()
    FLOAT32  = auto()
    FLOAT64  = auto()
    INT32    = auto()
    INT64    = auto()
    BOOL     = auto()


# 백엔드별 미지원 dtype — 사용 시 즉시 TypeError
_UNSUPPORTED: dict[str, set[DType]] = {
    "numpy": set(),
    "mlx":   {DType.FLOAT64, DType.INT64, DType.BFLOAT16},
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
        DType.BOOL:     mx.bool_,
        # FLOAT64, INT64 없음 — to_backend_dtype에서 에러
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
        # BFLOAT16 없음 — to_backend_dtype에서 에러
    }


_DTYPE_BUILDERS = {
    "numpy": _build_numpy_map,
    "mlx":   _build_mlx_map,
    "cupy":  _build_cupy_map,
}


def to_backend_dtype(dtype: DType) -> Any:
    """axon DType → 현재 백엔드의 실제 dtype 타입으로 변환.

    미지원 dtype은 폴백 없이 즉시 TypeError.
    """
    name = get_backend()
    if dtype in _UNSUPPORTED[name]:
        raise TypeError(
            f"{dtype.name} is not supported on the {name!r} backend. "
            f"Switch to a supported backend first: axon.set_backend('numpy')"
        )
    return _DTYPE_BUILDERS[name]()[dtype]
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

```
BackendProtocol 정의
    → NumpyBackend 구현
    → Tensor._data 타입을 Array로 추상화
    → Op들이 xp.xxx 호출로 변경
    → backward() in-place 분기 추가
    → 백엔드 동일 결과 테스트
    → MLXBackend 구현
    → lazy eval 처리
    → @axon.jit (mx.compile 위임)
    → CuPyBackend (옵션)
```

NumPy를 완전히 동작하게 만든 뒤 MLX를 붙인다.
Op 구현체는 `xp.xxx`만 호출하므로 백엔드 교체 시 수정 불필요.