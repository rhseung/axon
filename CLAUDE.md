# axon — 내부 설계 메모

axon 의 자료구조와 설계 결정을 적어두는 곳이에요. 사용자용은 `README.md`
에 따로 두고, 여기는 두 달 뒤의 제가 다시 펼쳐도 "왜 이렇게 했더라" 가
바로 떠오르도록 쓰는 게 목표예요.

## 0. 목표

자동미분 엔진을 만들고, 그 위에 신경망 프레임워크를 얹어서 GPT-2 124M
inference 까지 가는 거예요. HuggingFace 가중치 로드해서 logits 가
일치하면 성공이에요. 추상화는 그 단계에서 진짜 필요해질 때까지
미뤄둡시다.

이 프로젝트는 공부용 토이 프로젝트예요. PyPI 배포 안 하고, production
사용 안 하고, "이렇게 만들어봤어요" 의 기록이에요. 그래서 결정의 이유를
이렇게 길게 적어둘 수 있는 거고요.

---

## 1. 핵심 철학

### 1.1. 데이터와 그래프를 섞지 말기

자료가 다섯 레이어로 흘러가요:

```
DType                  타입 마커 (FLOAT16/32/64, BFLOAT16, INT32/64, BOOL)
  ↓
Array[D]               순수 데이터. 백엔드 ndarray 그 자체.
  ↓
xp                     백엔드 진입점. Array 만드는 길은 여기 하나.
  ↓
Constant[D] / Var[D]   그래프 노드. Constant 는 비추적 (입력 데이터),
                       Var 는 추적 (학습 weight 와 중간 결과).
  ↓
Net                    학습 weight 를 들고 있는 모듈. forward 정의.
```

여기서 가장 중요한 건 `Array` 가 데이터를 도맡는다는 점이에요. 모든 걸
한 타입에 욱여넣지 않아요. DataLoader 가 던져주는 배치, weight 의 초기값,
optimizer 의 내부 state, 체크포인트로 저장하는 값 — 다 Array 면 충분해요.
그래프 노드는 forward 가 도는 동안만 살아있고, backward 가 끝나면
그래프랑 같이 사라져요. 학습 weight 인 `Var` 만 Net 안에 영구 보존돼요.

### 1.2. ∂L/∂x 는 x 가 추적될 때만 존재해요

그래프 노드를 두 타입으로 나눴어요:

- **`Var`** — 추적 노드. `grad` 필드가 있고, `backward`, `step` 메서드도
  있어요. 학습 weight 와 중간 결과.
- **`Constant`** — 비추적 노드. `grad` 필드, `backward`, `step` 메서드
  전부 없어요. 입력 데이터, 비학습 상수, KV cache.

`Var` 와 `Constant` 가 별도 클래스라 정적 타입에서 잘못된 사용이 잡혀요:

```python
x = Constant(x_batch)
x.grad             # 정적 타입 에러: Constant 에 grad 없음
x.backward()       # 정적 타입 에러: Constant 에 backward 없음
```

수학 표기 $L = f(x; W, b)$ 에서 $x$ 는 상수, $W, b$ 는 변수 — 미분
대상이 정적 타입에 그대로 옮겨진 거예요.

### 1.3. 추상화는 막힐 때 들이기

- **연산자 오버로딩 (`x + y`, `x @ W`)** 은 v1 까지 미뤘어요. v0 는 명시
  함수 (`F.add(x, y)`) 만으로 backward 가 정확한지부터 확인합시다.
- **Named Dim** 은 형상 검증 도구예요. Transformer 디버깅이 진짜로
  버거워질 때 꺼냅시다.
- **Mixed precision, JIT, gradient checkpointing** 은 GPT-2 학습이
  메모리에 부딪힐 때 씁시다.

미리 깔아두면 코드만 무거워지고 검증이 안 돼요. 막혀봐야 그 추상화가
풀어주는 게 뭔지 손에 잡히거든요.

### 1.4. Parameter 는 Var 의 leaf 일 뿐

axon 엔 `Parameter` 같은 별도 클래스가 없어요. 학습 weight 는 `Var` 의
`_op is None` 인 인스턴스 (= leaf) 일 뿐이에요. `is_parameter` property
가 한 줄로 정의되고, optimizer 부착도 leaf 인 `Var` 에만 의미가 있어요.

```python
class Var(Node):
    _optimizer: Optimizer | None      # leaf 일 때만 의미

    @property
    def is_parameter(self) -> bool:
        """학습 leaf 인지 — _op is None 이면 leaf."""
        return self._op is None
```

마커 클래스 없이 그래프 metadata 만으로 학습 leaf 가 결정돼요.
PyTorch 의 `nn.Parameter` 처럼 따로 박스 만들 필요가 없어요.

### 1.5. Optimizer 는 weight 의 일부

PyTorch 처럼 optimizer 가 별도 객체로 떠다니지 않아요. 학습 weight 인
`Var` 가 자기 update 정책 (`Optimizer` 인스턴스) 을 직접 들고 있어요.
Adam 의 m, v 같은 state 도 그 weight 의 일부로 같이 살아요.

```python
W = Var(weight_init, optimizer=AdamW(lr=3e-4))
```

학습 루프는 두 줄:

```python
loss.backward()
loss.optimize()      # 그래프의 모든 학습 weight 가 자기 optimizer 로 update
```

`optimizer.zero_grad()` 매크로 호출 패턴이 사라져요. layer-wise LR,
weight-decay 분기, freeze 같은 흔한 워크로드도 자연스럽게 표현돼요.
자세한 건 §2.10 / §6.7 에서.

---

## 2. 자료구조

### 2.1. DType

```python
class DType:
    is_floating: bool = False
    is_integer:  bool = False

    FLOAT16:  type[DType]
    BFLOAT16: type[DType]
    FLOAT32:  type[DType]
    FLOAT64:  type[DType]
    INT32:    type[DType]
    INT64:    type[DType]
    BOOL:     type[DType]
```

백엔드 독립 dtype 마커예요. 인스턴스 안 만들고 클래스 자체를 쓰면 돼요
(`DType.FLOAT32`).

Enum 이 아니라 class 계층인 이유는 generic 자리에 바로 쓰고 싶어서예요.
`Array[DType.FLOAT32]` 가 그대로 valid type form 이 되거든요. Enum 이면
`Literal[DType.FLOAT32]` 로 한 겹 더 감싸야 했어요.

백엔드 native dtype 으로의 변환은 `axon.backend._dtype.to_backend_dtype`
이 맡아요. MLX 의 FLOAT64 처럼 백엔드가 지원 안 하는 조합은 폴백 없이
바로 `TypeError` — 조용히 다른 dtype 으로 갈아끼우면 디버깅이 지옥이라.

### 2.2. Array[D] — Protocol

```python
class Array[D: DType = DType](Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> Any: ...
    @property
    def T(self) -> Self: ...

    # 산술 / 비교 / 인덱싱 / 스칼라 변환 — numpy-like 풀 인터페이스
    def __add__(self, other: Self | int | float) -> Self: ...
    def __matmul__(self, other: Self) -> Self: ...
    def __getitem__(self, idx: Any) -> Self: ...
    # ... (전체는 axon/backend/protocol.py)
```

여기서 axon 만의 array 클래스를 따로 만들지 않은 게 핵심이에요.
`np.ndarray`, `mx.array`, `cp.ndarray` 가 이미 이 Protocol 을 구조적으로
만족하거든요. axon 의 "Array" 는 새 타입이 아니라 셋을 묶어주는 이름인
셈이에요.

`Self` 반환을 쓴 게 신기한 효과를 내요. `Self` 는 호출 시점의 구체
타입으로 specialize 되니까 — `np.ndarray + np.ndarray → np.ndarray`
처럼 — Protocol 만족하는 모든 타입이 자기 타입으로 닫혀요.

`D` 는 phantom marker 예요. runtime `.dtype` 은 백엔드 native (`np.dtype`,
`mx.Dtype` 등) 이고, `D` 는 정적 분석용 표시일 뿐이에요. axon `DType`
으로의 실제 변환은 `Var.dtype` / `Constant.dtype` 에서
`from_backend_dtype` 가 처리해요.

### 2.3. xp — 백엔드 진입점

```python
import axon.backend as xp

xp.exp(x)               # 현재 백엔드의 exp
xp.matmul(a, b)
xp.random.normal((n,))  # 난수
```

`_BackendProxy` 가 `__getattribute__` 로 매번 현재 백엔드한테 위임해요.
덕분에 `set_backend()` 후에도 같은 `xp` 변수가 새 백엔드를 가리키게
돼요 — import 다시 안 해도 돼요.

정적 타입은 `BackendProtocol` 로 잡혀 있어서 pyrefly 나 IDE 가 멤버를
다 추론해줘요. 런타임만 프록시인 거예요.

구성은 이렇게 돼 있어요:

- `BackendProtocol` — 백엔드 인스턴스 인터페이스. 생성, 수학, 축소,
  형상, 인덱싱, 변환을 한 곳에 모아둠.
- `RandomProtocol` — 난수 따로. `xp.random` 으로 접근.
- 구현체는 `NumpyBackend` (기준), `MLXBackend` (auto async_eval 처리),
  `CuPyBackend` (옵션).

### 2.4. Constant[D] / Var[D] — 그래프 노드

두 클래스가 공개 base `Node` 의 자식이에요. `Node` 자체는 user 가 *인스턴스화*
할 일은 없지만 (Constant / Var 만 만들고), forward 시그니처 같은 type annotation
자리에선 노출이 필요해요 — `Constant | Var` union 이 매 시그니처마다 verbose 한
것도 이유. PyTorch 의 `Tensor` / JAX 의 `Array` 와 같은 포지션이에요. "두 구체
타입의 공통 base" 로 type-only 공개.

```python
class Node[D: DType = DType]:
    """모든 그래프 노드의 공통 기반. type annotation 용 공개."""
    _data: Array[D]
    _op: Op | None
    _inputs: tuple[Node, ...]


class Constant[D: DType = DType](Node[D]):
    """비추적 그래프 노드. 입력 데이터, 비학습 상수, KV cache 등.
    grad / backward / step 없음."""

    def __init__(self, data, *, dtype=DType.FLOAT32):
        ...

    @classmethod
    def _from_op(cls, data, op, inputs) -> "Constant[D]":
        out = object.__new__(cls)
        out._data = data
        out._op = op
        out._inputs = inputs
        return out


class Var[D: DType = DType](Node[D]):
    """추적 그래프 노드. 학습 weight 또는 중간 결과.
    leaf (=학습 weight) 면 _optimizer 가 부착됨."""
    grad: Array[D]
    _optimizer: Optimizer | None      # leaf 일 때만 의미

    def __init__(
        self,
        data,
        *,
        dtype=DType.FLOAT32,
        optimizer: Optimizer | None = None,
    ):
        ...
        self.grad = xp.zeros_like(self._data)
        self._optimizer = optimizer

    @classmethod
    def _from_op(cls, data, op, inputs) -> "Var[D]":
        out = object.__new__(cls)
        out._data = data
        out._op = op
        out._inputs = inputs
        out.grad = xp.zeros_like(data)
        out._optimizer = None         # 중간 노드는 optimizer 안 가짐
        return out

    @property
    def is_parameter(self) -> bool:
        """학습 leaf 인지 — _op is None 이면 leaf."""
        return self._op is None

    def backward(self) -> None: ...
    def optimize(self) -> None: ...
```

세 가지 모습이 있어요:

| 어떤 노드냐 | 클래스 | `_op` | `_inputs` | `grad` | `_optimizer` |
|---|---|---|---|---|---|
| 입력 데이터 / 비학습 상수 | `Constant` | `None` | `()` | 없음 | 없음 |
| 비추적 중간 결과 (Constant 만 모은 chain) | `Constant` | Op | (...) | 없음 | 없음 |
| 학습 weight | `Var` | `None` | `()` | 있음 | 부착됨 |
| 추적 중간 결과 | `Var` | Op | (...) | 있음 | `None` |

`Constant` chain 은 거의 안 발생해요 — 보통 입력 데이터 (Constant) 와
학습 weight (Var) 의 chain 이라 결과가 `Var`. 하지만 inference 때 `Var`
없이 `Constant` 만 흐를 수 있어서, 이 케이스도 일관되게 처리.

`Op.apply` 가 입력 보고 어떤 클래스로 결과 만들지 dispatch 해요:

```python
class Op[D]:
    def apply(self, *inputs: Node[D]) -> Node[D]:
        self.validate(*inputs)
        out_array = self.forward(*(n._data for n in inputs))
        if any(isinstance(n, Var) for n in inputs):
            return Var._from_op(out_array, op=self, inputs=inputs)
        return Constant._from_op(out_array, op=self, inputs=inputs)
```

여기서 헷갈릴 만한 게 하나 있어요. "추적된다 (`Var`)" 와 "학습 leaf 다
(`is_parameter`)" 는 다른 개념이에요. 입력 `x: Constant` 와 weight
`W: Var` 의 곱 `x @ W` 는 `Var` 인데 학습 leaf 는 아니에요 — `W` 로 grad
가 흘러야 하니까 `Var` 는 맞지만, optimizer 가 업데이트할 weight 는
아니거든요. 그래서 `Var` 와 `is_parameter` 가 두 단계로 나뉘어 있는
거예요.

### 2.5. Op[D] — 연산 정의

```python
class Op[D: DType](ABC):
    @abstractmethod
    def forward(self, *inputs: Array[D]) -> Array[D]:
        """순전파 y = f(x_1, ..., x_n). 순수 Array 산수만."""
        ...

    @abstractmethod
    def backward(
        self, grad: Array[D], *inputs: Array[D]
    ) -> tuple[Array[D], ...]:
        """체인룰. 반환 tuple 길이는 inputs 길이와 같아야 해요."""
        ...

    def validate(self, *inputs: Node[D]) -> None:
        """입력이 이 Op 에 합당한지 검증. 기본은 no-op, 자식이 override."""
        pass

    def apply(self, *inputs: Node[D]) -> Node[D]:
        self.validate(*inputs)
        out_array = self.forward(*(n._data for n in inputs))
        if any(isinstance(n, Var) for n in inputs):
            return Var._from_op(out_array, op=self, inputs=inputs)
        return Constant._from_op(out_array, op=self, inputs=inputs)
```

**`validate`** 가 forward 전에 입력 shape / dtype / 개수가 합당한지
확인해요. backend 가 던지는 raw 에러 ("shapes (3,4) and (5,6) not
aligned") 보다 친절한 메시지로 잡혀요.

기본은 no-op 이고, 검증이 의미 있는 Op (MatMul, Reshape, Concat, Sum
with axis 등) 가 override 해요:

```python
class MatMul[D](Op[D]):
    def validate(self, *inputs: Node[D]) -> None:
        a, b = inputs
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError(
                f"MatMul requires ≥2D inputs, got {a.ndim}D and {b.ndim}D"
            )
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                f"MatMul shape mismatch: {a.shape} @ {b.shape} "
                f"(last dim {a.shape[-1]} ≠ second-to-last {b.shape[-2]})"
            )
```

단순 산술 (Add, Mul) 은 broadcast 가 알아서 해줘서 override 할 필요
없어요. 복잡한 Op 만 필요할 때 추가하면 돼요.

NamedDim 시스템 (미래) 은 이 위에 얹히는 별도 layer 예요. forward 안에서
사용자가 `x.dims("B T C")` 로 검증하는 — 그건 §7 미래 의제에서 다시
다뤄요.

---

`UnaryOp`, `BinaryOp` 로 안 나눴어요. 처음엔 나눴었는데 다시 보니 별로
얻는 게 없더라고요:

- 분리가 진짜 가치를 가지려면 base 가 공통 로직을 들고 있어야 해요.
  지금은 인자 unpack 한 줄이랑 tuple wrap 한 줄이 전부예요. 무게가 안
  나가요.
- 곧 ternary (`Where`) 랑 variadic (`Concat`) 도 들어올 거예요.
  분리해두면 `TernaryOp`, `VariadicOp` 가 끝없이 늘거나 일부 Op 만 `Op`
  직접 상속해서 일관성 깨져요.
- 자식 Op 첫 줄에 `(x,) = inputs` 나 `a, b = inputs` 가 들어가는데,
  이게 오히려 인자 개수를 코드 첫 줄에서 명시적으로 보여줘서 가독성에
  도움 돼요.

자식 Op 는 이렇게 생겼어요:

```python
class Add[D: DType](Op[D]):
    def forward(self, *inputs: Array[D]) -> Array[D]:
        a, b = inputs
        return a + b

    def backward(
        self, grad: Array[D], *inputs: Array[D]
    ) -> tuple[Array[D], ...]:
        return (grad, grad)


class PowConstExp[D: DType](Op[D]):
    """y = x ** n (n: Scalar). 지수가 상수라 backward 에 log 항이 없어요."""

    def __init__(self, n: Scalar):
        self.n = n

    def forward(self, *inputs: Array[D]) -> Array[D]:
        (x,) = inputs
        return x ** self.n

    def backward(
        self, grad: Array[D], *inputs: Array[D]
    ) -> tuple[Array[D], ...]:
        (x,) = inputs
        return (grad * self.n * x ** (self.n - 1),)
```

`PowConstExp` 의 `n` 처럼 학습 안 되는 메타 (axis, shape, mask, 상수
등) 는 그래프에 안 넣고 Op 인스턴스 필드로 들고 있어요. 그래프엔 학습
대상만 둬야 깔끔하거든요.

규칙은 이거예요: 상수의 backward 가 *위험한 산수* (log, sqrt, div) 를
부르면 별도 Op 으로 갈라요. 안전하면 Scalar lift 로 통합해요. Pow
패밀리는 전자예요 — `xp.log(a)` 가 a < 0 에서 NaN 이 터져버려요. Where
의 mask 는 후자고요.

반환 tuple 길이는 정적 타입으론 못 잡아요 (`tuple[Array, ...]` 가변).
대신 `Var.backward` 가 런타임에 잡아줘요:

```python
input_grads = v.op.backward(v.grad, *inputs)
assert len(input_grads) == len(v.inputs), (
    f"{type(v.op).__name__}.backward returned {len(input_grads)} grads "
    f"for {len(v.inputs)} inputs"
)
```

v0 에서 새 Op 추가하다가 실수하면 첫 테스트에서 바로 잡혀요.

### 2.6. Scalar

```python
Scalar = int | float
```

Python primitive 상수예요. `pow(x, 2)` 의 `2` 나 `PowConstBase(c=2.0)`
의 `c` 같은 거. Op 인스턴스 필드로 보관하거나 functional dispatch 에서
써요.

`isinstance(x, Scalar)` 가 동작해야 해서 평범한 union 으로 뒀어요. PEP
695 `type` alias 였으면 isinstance 가 안 되거든요.

### 2.7. functional — 사용자 API

```python
import axon.functional as F

F.add(a, b)
F.matmul(a, b)
F.relu(x)
```

**`functional` 이 사용자가 보는 유일한 연산 진입점이에요.** Op 클래스
자체는 비공개로 둬요 — `axon.__init__` 에서 export 안 하고, 사용자가
`Add().apply(a, b)` 같은 식으로 직접 호출하지 않도록.

이유는 두 가지:

1. 같은 일을 두 길로 할 수 있으면 어느 게 표준인지 헷갈려요. functional
   하나로 일원화.
2. Op 시그니처는 내부 추상화라 바뀔 수 있어요 (`validate` 추가, 인자
   순서 조정 등). 사용자한테 노출되지 않으면 바꾸기 자유로워요.

`axon.operation` 패키지 자체는 내부 디렉토리로 남고, 패키지 root 에서
re-export 안 하는 정도로 충분해요. 깊은 import path (`axon.operation.add`)
는 어차피 내부 코드만 만져요.

#### Overload 로 반환 타입 좁히기

functional 함수가 `Constant` / `Var` 인자에 따라 반환 타입을 정확히
추론해요. overload 로 정의:

```python
@overload
def matmul[D](a: Var[D], b: Var[D] | Constant[D]) -> Var[D]: ...
@overload
def matmul[D](a: Constant[D], b: Var[D]) -> Var[D]: ...
@overload
def matmul[D](a: Constant[D], b: Constant[D]) -> Constant[D]: ...
def matmul(a, b):
    return MatMul().apply(a, b)
```

규칙: 인자 중 하나라도 `Var` 면 반환은 `Var`. 둘 다 `Constant` 면 반환은
`Constant`.

사용자한텐 overload 가 안 보이고, 그냥 `loss = F.cross_entropy(logits,
y)` 호출하면 `loss: Var` 로 자동 추론. `loss.backward()` 가 정적 타입에서
허용됨.

#### v1 — 연산자 오버로딩

v0 는 명시 함수만, v1 에 dunder 도입. Scalar lifting 은 `_coerce_pair`
헬퍼:

```python
def _coerce_pair[D](
    a: Node[D] | Scalar,
    b: Node[D] | Scalar,
) -> tuple[Node[D], Node[D]]:
    """둘 중 적어도 하나는 Node 여야 해요. Scalar 는 상대 dtype 으로 lift."""
    match a, b:
        case Node(), Node():
            return a, b
        case Node(), _:
            return a, Constant(b, dtype=a.dtype)
        case _, Node():
            return Constant(a, dtype=b.dtype), b
        case _:
            raise TypeError(...)


def add(a, b):
    a, b = _coerce_pair(a, b)
    return Add().apply(a, b)
```

Scalar 는 `Constant` 로 lift — 미분 안 되는 상수니까 자연스러움. backend
Array 자동 lift 는 일부러 안 해요 — 백엔드마다 dispatch 우선순위
매커니즘이 달라서 (`__array_priority__` 는 NumPy 만 보거든요), 자동으로
받아들이면 조용한 버그가 잘 생겨요. 사용자가 명시적으로
`Constant(arr)` 로 감싸는 게 안전해요.

연산자 오버로딩이 v1 에 도입되면 dunder 는 한 줄짜리 위임이 돼요:

```python
class Node:
    __array_priority__ = 1000   # NumPy interop

    def __add__(self, other): return F.add(self, other)
    def __radd__(self, other): return F.add(other, self)
    # ... 13 개 dunder 가 1:1 functional 위임
```

`pow` 만 다른 binary 들이랑 패턴이 살짝 달라요:

```python
def pow(a, b):
    if isinstance(b, Scalar):
        return PowConstExp(b).apply(a)
    if isinstance(a, Scalar):
        return PowConstBase(a).apply(b)
    return Pow().apply(a, b)
```

수치 안정성 때문이에요. `Pow.backward` 의 `xp.log(a)` 항을 우회하려고
const 분기를 별도 Op 으로 빼둔 거예요.

#### functional vs net 의 경계

`functional` 은 state 가 없는 순수 연산만. layer (Linear, LayerNorm 등)
나 옵션 보관이 필요한 loss (CrossEntropyLoss 의 `ignore_index` 등) 는
`axon.net` 에 module 형태로. 활성화 함수처럼 양쪽 다 의미 있는 건
양쪽에 둠 (§2.10 끝에서 자세히).

### 2.8. Var.backward 와 Var.optimize

```python
class Var[D]:
    def backward(self) -> None:
        """그래프 따라 grad 누적. 누적 모드 — 매번 호출하면 grad 가 쌓임.
        새 step 시작할 때 zero 는 optimize() 또는 model.zero_grad() 가 처리."""
        self.grad = xp.ones_like(self._data)

        for n in _topological_order(self):
            if not isinstance(n, Var) or n._op is None:
                continue
            inputs = tuple(inp._data for inp in n._inputs)
            input_grads = n._op.backward(n.grad, *inputs)
            assert len(input_grads) == len(n._inputs), (
                f"{type(n._op).__name__}.backward returned "
                f"{len(input_grads)} grads for {len(n._inputs)} inputs"
            )
            for inp, inp_grad in zip(n._inputs, input_grads):
                if isinstance(inp, Var):
                    inp.grad = inp.grad + inp_grad
                # Constant 면 grad 흘려보낼 곳 없음 — skip

    def optimize(self) -> None:
        """그래프 도달 가능한 학습 weight 의 update + grad zero.
        backward 가 이미 호출됐다고 가정."""
        for n in _topological_order(self):
            if (isinstance(n, Var)
                and n.is_parameter
                and n._optimizer is not None):
                n._optimizer.update(n)
                n.grad = xp.zeros_like(n._data)
```

`Constant` 엔 `backward` / `step` 메서드가 없어요. 정적 타입에서
`x.backward()` (x 가 Constant) 가 에러로 잡혀요.

dict 안 써요. `Var` 는 다 `grad` 필드를 가지니까 거기 직접 누적해버려요.
중간 노드의 grad 메모리 걱정도 안 해도 돼요. 매 step 그래프가 새로
빌드되고, backward 가 끝나면 그래프 전체가 호출 컨텍스트 밖에서 GC 돼요.
중간 `Var` 의 grad 도 같이 사라져요. 살아남는 건 사용자 (Net) 가 들고
있는 학습 leaf 뿐이에요.

부수 효과로 backward 직후 ~ GC 전 사이엔 중간 노드의 grad 가 살아있어서
디버깅할 때 `intermediate_var.grad` 를 직접 들여다볼 수 있어요. 따로
flag 안 켜도 default 로 그렇게 동작해요.

**Gradient accumulation** 은 `step` 만 빼고 `backward` 를 여러 번:

```python
for micro in micro_batches:
    compute_loss(micro).backward()    # grad 누적만
final_loss.optimize()                      # 마지막에 한 번만 update + zero
```

`backward` 가 grad 를 zero 안 하는 건 의도된 동작 — accumulation 이
default. 명시적 zero 가 필요하면 `model.zero_grad()`.

### 2.9. Optimizer — weight 의 일부

PyTorch 의 `optim.Optimizer` 는 model 의 모든 parameter 를 받아 한 번에
관리하는 객체예요. axon 은 다르게 가요:

```python
class Optimizer(ABC):
    """단일 Var 의 update 정책 + state.
    Adam 의 m, v 같은 per-weight state 가 여기 살아요."""

    @abstractmethod
    def update(self, var: Var) -> None:
        """var._data 를 var.grad 로 업데이트. var._data 직접 수정."""
        ...


class SGD(Optimizer):
    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocity: Array | None = None

    def update(self, var: Var) -> None:
        if self.momentum > 0:
            if self._velocity is None:
                self._velocity = xp.zeros_like(var._data)
            self._velocity = self.momentum * self._velocity + var.grad
            var._data -= self.lr * self._velocity
        else:
            var._data -= self.lr * var.grad


class AdamW(Optimizer):
    def __init__(
        self,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self._m: Array | None = None
        self._v: Array | None = None
        self._t: int = 0

    def update(self, var: Var) -> None:
        if self._m is None:
            self._m = xp.zeros_like(var._data)
            self._v = xp.zeros_like(var._data)
        self._t += 1
        b1, b2 = self.betas
        self._m = b1 * self._m + (1 - b1) * var.grad
        self._v = b2 * self._v + (1 - b2) * var.grad ** 2
        m_hat = self._m / (1 - b1 ** self._t)
        v_hat = self._v / (1 - b2 ** self._t)
        var._data -= self.lr * (m_hat / (xp.sqrt(v_hat) + self.eps)
                                + self.weight_decay * var._data)
```

핵심: **한 `Optimizer` 인스턴스는 한 `Var` 를 위한 거예요.** 같은
`AdamW(lr=3e-4)` 인스턴스를 여러 Var 가 공유하면 `_m`, `_v` 가 섞여 망가져요.
각 Var 가 자기 인스턴스를 가져야 해요. `Net.optimizer` default 매커니즘이
자동으로 deepcopy 해서 부착해요 (§2.10).

State 가 weight 옆에 사는 게 정직해요 — Adam 의 m, v 는 그 weight 의
이력이지 별도 객체가 아니거든요. weight 가 GC 되면 state 도 자동 GC.

### 2.10. Net

```python
class Net(ABC):
    optimizer: Optimizer | None = None      # class attribute, 사용자가 override

    def __setattr__(self, name: str, value: Any) -> None:
        # 학습 weight 가 들어오면 default optimizer 의 *복제* 를 부착
        if (isinstance(value, Var)
            and value.is_parameter
            and value._optimizer is None
            and type(self).optimizer is not None):
            value._optimizer = copy.deepcopy(type(self).optimizer)
        super().__setattr__(name, value)

    @abstractmethod
    def forward(self, x: Node) -> Var: ...

    def parameters(self) -> list[Var]:
        out: list[Var] = []
        visited: set[int] = set()

        def collect(value):
            if id(value) in visited:
                return
            visited.add(id(value))

            if isinstance(value, Var) and value.is_parameter:
                out.append(value)
                return
            if isinstance(value, Net):
                for child in value.__dict__.values():
                    collect(child)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        collect(self)
        return out

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = xp.zeros_like(p._data)

    def clip_grad_norm(self, max_norm: float) -> Array:
        """모든 학습 weight 의 grad 를 합친 L2 norm 이 max_norm 넘지 않게 정규화.
        반환은 clipping 전 norm — 모니터링용."""
        params = self.parameters()
        total_norm_sq = sum(xp.sum(p.grad ** 2) for p in params)
        total_norm = xp.sqrt(total_norm_sq)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in params:
                p.grad = p.grad * clip_coef
        return total_norm

    def grad_norm(self) -> Array:
        """모니터링용 grad norm 측정."""
        params = self.parameters()
        return xp.sqrt(sum(xp.sum(p.grad ** 2) for p in params))
```

학습 weight 를 들고 있고 forward 를 정의하는 자리예요.

세 가지 중요한 매커니즘:

1. **`Net.optimizer` class attribute** — 자식 클래스가 override 해서
   default 지정. `MLP.optimizer = AdamW(lr=3e-4)` 한 줄.

2. **`__setattr__` 자동 부착** — weight `Var` 가 attribute 로 들어올 때
   자동으로 `Net.optimizer` 의 deepcopy 를 부착. 사용자가 매번
   `optimizer=AdamW(...)` 안 써도 됨. 명시 override 하고 싶으면
   `Var(..., optimizer=다른_거)` 로 생성하면 자동 부착이 skip.

3. **`parameters()` 자동 수집** — `__dict__` 재귀 순회로 학습 leaf 들
   모음. 별도 등록 매커니즘 없음. `is_parameter` (= `Var` 이고 `_op
   is None`) 로 판단.

4. **Global 동작은 Net 메서드** — `clip_grad_norm`, `grad_norm`,
   `zero_grad`. 모든 weight 를 한 번에 다루는 동작은 여기. optimizer 와
   직교한 차원이라 깔끔.

#### axon.net 패키지 구조

`axon.net` 은 base `Net` 클래스 + 자주 쓰이는 구체 module 을 같이
두는 곳이에요 (PyTorch `nn` 패키지와 같은 패턴):

```
axon/net/
├── __init__.py        # Net + 모든 구체 module export
├── net.py             # base Net 클래스
├── linear.py          # Linear
├── activation.py      # ReLU, GELU, SiLU, Tanh, Sigmoid
├── norm.py            # LayerNorm, RMSNorm, BatchNorm
├── dropout.py         # Dropout
├── embedding.py       # Embedding
├── loss.py            # CrossEntropyLoss, MSELoss, BCELoss
├── sequential.py      # Sequential, ModuleList, ModuleDict
├── attention.py       # MultiHeadAttention
└── transformer.py     # TransformerBlock 등
```

사용자는 `import axon.net as net` 으로 짧게 alias:

```python
import axon.net as net

class MLP(net.Net):
    def __init__(self):
        self.fc1 = net.Linear(784, 128)
        self.fc2 = net.Linear(128, 10)
```

#### functional vs net 의 경계

| | functional | net |
|---|---|---|
| state | 없음 | 있을 수도 (weight, 옵션) |
| 호출 | 함수 | 인스턴스 메서드 |
| 위치 | forward 안 inline | `__init__` 에서 인스턴스화 |

활성화 함수와 loss 는 양쪽 다 둬요. 같은 동작인데 사용 패턴이 둘:

```python
# functional — forward 안 inline
def forward(self, x):
    h = F.relu(self.fc1(x))
    return self.fc2(h)

# net — Sequential 에 끼우거나 옵션 보관
self.layers = net.Sequential(
    net.Linear(784, 128),
    net.ReLU(),
    net.Linear(128, 10),
)

loss_fn = net.CrossEntropyLoss(ignore_index=-100)   # 옵션 인스턴스에
```

`net` 의 활성화 module 은 functional 의 thin wrapper:

```python
class ReLU(Net):
    def forward(self, x):
        return F.relu(x)
```

State 가지는 활성화 (PReLU 등) 는 의미 있게 module — `__init__` 에서
weight Var 만들고 forward 에서 사용. functional 한 곳에 모은 lifting
매커니즘 (§2.7) 과 별개로, layer 의 state 는 module 인스턴스에 살아요.

Linear / LayerNorm / Embedding 같은 weight 가지는 layer 는 functional 형태
없음 — state 를 인스턴스 밖에 두면 매번 인자로 받아야 해서 코드만
무거워져요.

### 2.11. 학습 루프 — 두 줄 패턴

```python
import axon.net as net
import axon.functional as F

class MLP(net.Net):
    optimizer = AdamW(lr=3e-4)

    def __init__(self):
        self.fc1 = net.Linear(784, 128)
        self.fc2 = net.Linear(128, 10)

    def forward(self, x: Constant) -> Var:
        h = F.relu(self.fc1(x))
        return self.fc2(h)


model = MLP()
loss_fn = net.CrossEntropyLoss()

for x_batch, y_batch in loader:
    logits = model.forward(Constant(x_batch))
    loss = loss_fn(logits, Constant(y_batch))

    loss.backward()
    loss.optimize()
```

`optimizer.zero_grad()` 호출이 사라졌어요. `optimize()` 안에서 update 직후
바로 grad zero 하니까. weight 도 `Var` 로 직접 다루지 않고 `net.Linear`
가 안에서 만들어둠 — 사용자는 layer 만 선언하면 됨.

Layer-wise LR (LLRD) 도 자연스러움:

```python
class TransformerBlock(net.Net):
    def __init__(self, dim, *, lr_scale: float = 1.0):
        opt = AdamW(lr=3e-4 * lr_scale, weight_decay=0.01)
        opt_no_decay = AdamW(lr=3e-4 * lr_scale, weight_decay=0.0)

        # Linear 가 자기 W 와 b 에 다른 optimizer 부착하도록 인자로 전달
        self.attn = net.MultiHeadAttention(
            dim,
            weight_optimizer=opt,
            bias_optimizer=opt_no_decay,
        )
        self.ln = net.LayerNorm(dim, optimizer=opt_no_decay)


class GPT(net.Net):
    def __init__(self, n_layers=12):
        self.blocks = [
            TransformerBlock(dim=768, lr_scale=0.9 ** (n_layers - i))
            for i in range(n_layers)
        ]
```

PyTorch 의 `param_groups` 패턴이 layer 클래스 안으로 자연스럽게
녹아들어가요.

### 2.12. 자료구조 한눈에 보기

```
DType
  ↓
Array[D] (Protocol)
  ├── np.ndarray, mx.array, cp.ndarray 가 만족
  └── BackendProtocol (xp)
       ├── NumpyBackend
       ├── MLXBackend
       └── CuPyBackend

Node[D] (공개 base — type annotation 용, 직접 인스턴스화 X)
  ├── _data: Array[D]
  ├── _op: Op | None
  └── _inputs: tuple[Node, ...]

Constant[D] (Node)
  └── 비추적. grad / backward / step 없음.

Var[D] (Node)
  ├── grad: Array[D]
  ├── _optimizer: Optimizer | None
  ├── is_parameter: bool                # property
  ├── backward()                        # grad 누적
  └── optimize()                        # 도달 weight update + zero

Op[D] (ABC, 비공개)
  ├── forward(*Array) -> Array
  ├── backward(grad, *Array) -> tuple[Array, ...]
  ├── validate(*Node) -> None
  └── apply(*Node) -> Node            # 입력에 Var 있으면 Var, 없으면 Constant

functional (공개 API)
  ├── _coerce_pair (v1)
  └── add, sub, mul, div, matmul, neg, pow, ...
        └── 각 함수 overload 3 개 (Var/Constant 분기)

Optimizer (ABC)
  └── update(Var) -> None               # Var._data 직접 수정
       ├── SGD
       └── AdamW

Net (ABC)
  ├── optimizer: Optimizer | None       # class attribute, default
  ├── __setattr__                       # weight 에 default optimizer 자동 deepcopy
  ├── forward(x: Node) -> Var
  ├── parameters() -> list[Var]
  ├── zero_grad()
  ├── clip_grad_norm(max_norm) -> Array
  └── grad_norm() -> Array

axon.net (패키지)
  ├── Net                               # base
  ├── Linear, Embedding
  ├── LayerNorm, RMSNorm, BatchNorm, Dropout
  ├── ReLU, GELU, SiLU, Tanh, Sigmoid   # functional 의 module wrapper
  ├── CrossEntropyLoss, MSELoss, BCELoss
  ├── Sequential, ModuleList, ModuleDict
  └── MultiHeadAttention, TransformerBlock

Scalar = int | float
```

---

## 3. 백엔드 추상화

axon 은 NumPy / MLX / CuPy 셋을 동일한 인터페이스 뒤로 숨겨요. 사용자
코드가 한 줄도 안 바뀌고 백엔드를 갈아끼울 수 있는 게 목표예요.

### 3.1. 두 개의 Protocol

```python
class Array[D: DType = DType](Protocol):       # ndarray 의 공통 인터페이스
    shape, ndim, dtype, T
    __add__, __mul__, __matmul__, __getitem__, ...

class BackendProtocol(Protocol):                # 백엔드 인스턴스의 인터페이스
    name: str
    random: RandomProtocol
    array, zeros, ones, ...                     # 생성
    exp, log, sqrt, ...                         # 수학
    sum, mean, max, ...                         # 축소
    reshape, transpose, ...                     # 형상
    matmul, einsum                              # 선형대수
    where, take, tril, ...                      # 인덱싱
    to_numpy, from_numpy, eval, async_eval      # 변환 / 평가
```

`Array` 는 ndarray 객체 자체가 직접 지원하는 거 (연산자, 속성, 인덱싱).
`BackendProtocol` 은 객체만으론 표현 어려운 거 (생성, 축소, 형상 변환).
역할이 갈려요.

Op 구현체는 이렇게 두 길을 섞어 써요:

```python
class Add(Op):
    def forward(self, *inputs):
        a, b = inputs
        return a + b               # Array.__add__ 직접 사용

class Sum(Op):
    def forward(self, *inputs):
        (x,) = inputs
        return xp.sum(x, axis=self.axis)   # BackendProtocol 경유
```

### 3.2. 백엔드별 차이

| 항목 | NumPy | MLX | CuPy |
|---|---|---|---|
| 실행 | Eager | Lazy (eval 필요) | Eager |
| In-place | `+=`, `__setitem__` 지원 | 0.31+ 부터 native 지원 | 지원 |
| FLOAT64 | ✓ | ✗ (GPU 미지원) | ✓ |
| BFLOAT16 | float32 로 대체 | ✓ | ✗ |
| compile | 없음 | `mx.compile` | 없음 |

**MLX 의 lazy 처리** 가 가장 까다로웠어요. `mx.eval()` 호출 전엔 계산이
실제로 안 돌아요. axon 은 이걸 사용자 / framework 어느 레이어에도
노출하지 않으려고 reduction / argmax 류 op 들에 자동 `mx.async_eval` 을
걸어요:

```python
_AUTO_AE_METHODS = {
    "sum", "mean", "var", "max", "min", "norm",
    "argmax", "argmin",
}
```

이 op 들은 보통 chain 의 끝 (loss, accuracy, gradient norm) 에 위치해서,
거기서 async_eval 을 걸면 그 경로의 fusion 기회를 살리면서도 다음
reduction 까지는 lazy 가 유지돼요. 모든 op 에 async_eval 걸면 fusion
기회를 잃어서 NumPy 보다 느려져요 — 실측으로 검증한 결과예요.

### 3.3. DType 매핑

axon `DType` ↔ 백엔드 native dtype 변환은 `_dtype.py` 의 매핑 테이블이
맡아요. 미지원 조합은 폴백 없이 즉시 `TypeError`:

```python
_UNSUPPORTED: dict[str, set[type[DType]]] = {
    "numpy": set(),
    "mlx":   {DType.FLOAT64},     # GPU 미지원
    "cupy":  {DType.BFLOAT16},
}
```

조용히 다른 dtype 으로 갈아끼우면 결과가 미세하게 달라져 디버깅이
지옥이에요. numerical gradient check 처럼 FLOAT64 가 꼭 필요한 경우는
사용자가 명시적으로 NumPy 백엔드로 전환해야 해요.

### 3.4. backend 전환

```python
import axon
axon.set_backend("mlx")              # 한 줄로 전환
import axon.backend as xp            # 같은 xp 가 새 백엔드를 가리킴
```

`xp` 는 `_BackendProxy` 인스턴스라 `__getattribute__` 로 매번 현재 백엔드에
위임해요. `set_backend()` 후에도 같은 변수가 유효 — re-import 안 해도
돼요. 정적 타입은 `BackendProtocol` 로 찍혀 있어서 IDE / pyrefly 가 멤버
추론을 다 해줘요.

### 3.5. 검증

백엔드 추상화의 정합성은 **NumPy ↔ MLX parity 테스트** 로 잡아요. 같은
입력에 대해 두 백엔드 결과가 1e-5 안쪽으로 일치하는지 확인. 모든 Op /
forward / backward 가 대상이에요.

```python
def test_matmul_parity():
    x = np.random.randn(4, 3).astype(np.float32)
    w = np.random.randn(3, 5).astype(np.float32)

    axon.set_backend("numpy")
    out_np = (Var(x) @ Var(w)).as_numpy()

    axon.set_backend("mlx")
    out_mlx = (Var(x) @ Var(w)).as_numpy()

    np.testing.assert_allclose(out_np, out_mlx, atol=1e-5)
```

CuPy 는 로컬 (Apple Silicon) 에서 검증 불가라 `pytest -m cuda` 로
분리되어 있어요. Colab / Windows 환경에서 따로 돌려야 해요.

### 3.6. 지금 시점 완성도

백엔드 추상화는 v0 시점에 이미 완성되어 있어요. NumpyBackend,
MLXBackend, CuPyBackend 셋 다 구현됐고, `set_backend("mlx")` 로 즉시
전환 가능해요. 단계별 로드맵 (§4) 의 어느 v? 에서든 백엔드는 이미
주어진 인프라이고, 각 단계는 그 위에서 어떤 모델까지 학습되느냐의
진척도로만 보면 돼요.

---

## 4. 단계별 로드맵

각 단계마다 추가되는 자료구조 / 연산 / 검증 산출물을 적어둬요. 다음
단계로 넘어가기 전에 통과해야 하는 게이트 (= 검증 산출물) 가 있어요.

### v0 — 자동미분 엔진 검증 (현재 위치)

**자료구조**: `DType`, `Array` Protocol, `xp`, `Node` / `Constant` / `Var`,
`Op`, `functional`, `Var.backward`. `Net` / `Optimizer` 는 아직 없어요.

**Op**:
- 이항: `Add`, `Sub` (= `Add` + `Neg`), `Mul`, `Div`, `MatMul`, `Pow`
- 단항: `Neg`, `PowConstExp`, `PowConstBase`

**검증**:
- 각 Op 마다 numerical gradient check 통과 (모든 backward 가
  $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ 와 1e-5 안쪽 일치).
- `examples/pure_mnist.py` 의 forward + backward 가 axon 으로도 동작
  (Net 없이 손으로 chain 만들어서). pure 코드와 logits 일치.

**게이트**: 위 둘 통과. `Constant` / `Var` 분리 완료.

### v1 — 모델 학습 인프라

**자료구조 추가**: `Net`, `Optimizer` (`SGD`, `AdamW`), `Var.optimize`.
`Net.optimizer` class attribute + `__setattr__` 자동 부착.

**Op 추가**: `Sum`, `Mean`, `Maximum`, `LogSoftmax`.

**`axon.functional` 추가**: `relu`, `cross_entropy`, `mse_loss`, `log_softmax`,
`softmax` (LogSoftmax 위에서 동작).

**`axon.net` 신설**: `Net` base 외에 `Linear`, `ReLU`, `Sequential`,
`CrossEntropyLoss` 등 MNIST 학습에 필요한 최소 module.

**연산자 오버로딩 도입**: `_coerce_pair` 헬퍼와 함께 13 개 dunder.
사용자 코드가 `F.add(F.matmul(x, W), b)` 에서 `x @ W + b` 로 짧아져요.

**Functional overload 도입**: `Var` / `Constant` 분기 정적 타입 추론.

**Net 글로벌 동작**: `parameters()`, `zero_grad()`, `clip_grad_norm()`,
`grad_norm()`.

**검증**:
- MNIST val acc 97% 이상.
- PyTorch 같은 hyperparameter 셋업 대비 ±1% 안쪽 (parity check).

**게이트**: MNIST 학습 안정적으로 수렴. parity check 통과. `loss.backward()
loss.optimize()` 두 줄 패턴 정착.

### v2 — 형상 / 정규화

**Op 추가**: `Reshape`, `Transpose`, `Squeeze`, `Expand`, `Max`,
`GetItem`, `Where`, `Concat`.

**활성화 (functional 조합)**: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`.
별도 Op 으로 안 만들고 primitive 조합으로.

**`axon.net` 추가**: `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`,
`ReLU` / `GELU` / `SiLU` (functional wrapper), `Sequential`.

**검증**:
- 작은 LSTM 으로 TinyShakespeare 학습. loss 수렴 곡선이 PyTorch 구현과
  유사.
- `_unbroadcast` 헬퍼 검증 (broadcasting 후 grad 복원).

**게이트**: TinyShakespeare 학습 후 그럴듯한 셰익스피어풍 텍스트 생성.

### v3 — Attention + Transformer

**Op 추가**: `Gather`, `Sin`, `Cos`.

**`axon.net` 추가**:
- `ScaledDotProductAttention`, `MultiHeadAttention`
- Causal mask, Padding mask
- Sinusoidal PE, Learned PE, RoPE
- `TransformerBlock` (decoder-only, GPT 스타일)

**KV cache**: 추론 시 이전 k/v 재사용. `Constant` 로 두면 자연스러움 —
미분 안 되는 이력이니까.

**Tokenizer**: character-level (TinyShakespeare 용), Byte-level BPE
(GPT-2 호환).

**Sampling**: greedy, temperature, top-k, top-p.

**검증**: TinyShakespeare GPT 학습. loss ~1.5. 생성 텍스트 그럴듯함.

**게이트**: Transformer 가 cls-level 학습 가능. KV cache 활용한 incremental
decoding 동작.

### v4 — GPT-2 가중치 로드

**자료구조 추가**: `state_dict()`, `load_state_dict()`. axon 자체 포맷
(numpy 로 디렉토리 저장) + HuggingFace 키 매핑. Optimizer state 도 weight
와 함께 직렬화 (§2.9 참고).

**검증**:
- HF GPT-2 124M 가중치 로드 후 같은 입력에 대해 logits 차이 < 1e-4.
- greedy / top-p 생성 결과가 HF 와 일치.

**게이트**: 위 둘 통과. axon 의 자동미분 + 백엔드 추상화 + Transformer
구현이 production-grade reference 와 동등함을 증명.

### v5 — Stretch

여기부턴 GPT-2 from-scratch 학습이 목표예요. 메모리와 속도 압력이
본격적으로 들어오는 단계예요.

**추가 기능**:
- Mixed precision (bfloat16 forward + fp32 master).
- Gradient accumulation (이미 v1 부터 가능, 여기선 활용).
- Gradient checkpointing.
- Learning rate scheduler (CosineAnnealing, LinearWarmup).

**시도 (성공 보장 X)**:
- FineWeb 일부 또는 OpenWebText 일부로 GPT-2 124M from-scratch 학습.
- M 시리즈 한 대로는 시간 / 메모리 빠듯함. "합리적인 loss 도달" 이 목표.

**검증**: parity check 보다는 학습이 발산하지 않고 합리적 곡선을
그리는지 확인.

### 어디까지 와있나

지금 코드 베이스는 **v0 의 후반** 이에요:

- ✅ DType, Array Protocol, xp, BackendProtocol — 셋 백엔드 다 구현
- ✅ Op 추상화 (Add/Sub/Mul/Div/MatMul/Neg/Pow 패밀리)
- ✅ Net.parameters() 자동 수집
- ✅ Optimizer base
- ⏳ Node → Var / Constant 분리 마이그레이션
- ⏳ Parameter 클래스 제거
- ⏳ numerical gradient check
- ⏳ pure_mnist 를 axon 으로 forward + backward
- ⏳ Var.optimize + Net.optimizer 자동 부착 매커니즘
- ⏳ SGD / AdamW 구현 완성

위 ⏳ 들 끝나면 v1 (Net + 연산자 오버로딩 + MNIST 학습) 으로 넘어가요.

---

## 5. Op 카탈로그

(v? 표시는 그 단계에서 도입)

### v0 — 자동미분 검증용

| Op | forward | backward |
|---|---|---|
| `Add` (v0) | $a + b$ | $(g, g)$ |
| `Mul` (v0) | $a \cdot b$ | $(g \cdot b,\ g \cdot a)$ |
| `Div` (v0) | $a / b$ | $(g / b,\ -g \cdot a / b^2)$ |
| `MatMul` (v0) | $A @ B$ | $(g @ B^T,\ A^T @ g)$ |
| `Neg` (v0) | $-x$ | $(-g,)$ |
| `Sub` (v0) | $a - b$ | $(g, -g)$ — 또는 `Add` + `Neg` 조합 |
| `Pow` (v0) | $a^b$ | $(g \cdot b \cdot a^{b-1},\ g \cdot a^b \cdot \log a)$ |
| `PowConstExp` (v0) | $x^n$ | $(g \cdot n \cdot x^{n-1},)$ — log 회피 |
| `PowConstBase` (v0) | $c^x$ | $(g \cdot c^x \cdot \log c,)$ — c > 0 가정 |

### v1 — 모델 학습용

| Op | forward | backward |
|---|---|---|
| `Sum` (v1) | $\sum x$ (axis 따라) | broadcast 로 $g$ 복원 |
| `Mean` (v1) | $\bar{x}$ | $g / N$ broadcast |
| `Maximum` (v1) | $\max(a, b)$ elementwise | mask 로 $g$ 분배 |
| `LogSoftmax` (v1) | $x - \log \sum e^x$ | $g - e^{x'} \sum g$ |

### v2 — 형상 / 인덱싱

| Op | forward | backward |
|---|---|---|
| `Reshape` (v2) | view 변경 | grad 도 reshape |
| `Transpose` (v2) | 축 교환 | 같은 축 교환으로 복원 |
| `Squeeze` / `Expand` (v2) | 1-차원 추가/제거 | 반대 작업 |
| `Max` (v2) | $\max x$ (axis) | argmax 위치에만 grad |
| `GetItem` (v2) | `x[idx]` | scatter 로 0-init grad 채움 |
| `Where` (v2) | `cond ? a : b` | mask 로 grad 분배 |
| `Concat` (v2) | axis 따라 결합 | 같은 axis 따라 split |

### v3 — Attention 전용

| Op | forward | backward |
|---|---|---|
| `Gather` (v3) | indices 로 lookup | scatter |
| `Sin` / `Cos` (v3) | $\sin x, \cos x$ | $g \cos x, -g \sin x$ |

backward 의 broadcasting 처리 (a, b 의 shape 가 다를 때 grad 가 차원
줄어드는 거) 는 `_unbroadcast` helper 로 통합. `Mul.backward` 가 호출
직전에 `_unbroadcast(g * b, a.shape)` 같은 식으로.

---

## 6. Decision Log

설계 결정의 이유를 모아둬요. 두 달 뒤의 제가 "왜 이렇게 했더라" 하지
않도록.

### 6.1. 데이터와 그래프 분리 — Array vs 그래프 노드

**결정**: 데이터 (`Array`) 와 그래프 노드 (`Constant` / `Var`) 를 다른
타입으로.

**대안**: PyTorch 처럼 `Tensor` 한 클래스에 둘 다 욱여넣기.

**이유**:
- DataLoader 가 던져주는 배치, optimizer 의 state, weight 초기값,
  체크포인트 — 다 `Array` 면 충분. 그래프 정보가 필요한 건 forward 도는
  순간뿐.
- `Tensor` 한 클래스로 통합하면 매 객체가 grad / requires_grad / op /
  inputs 를 들고 있어야 함. 데이터로만 쓰는 99% 의 경우에 낭비.

### 6.2. ∂L/∂x 는 x 가 추적될 때만 — Constant / Var 분리

**결정**: 그래프 노드를 두 클래스로 분리. `Constant` 는 비추적 (grad,
backward, step 메서드 전부 없음), `Var` 는 추적.

**대안 1**: PyTorch 처럼 한 클래스 + `requires_grad: bool` 플래그.
**대안 2**: `Var` 한 클래스 + grad 필드를 property 로 친절한 에러.

**이유**:
- 수학 표기 $L = f(x; W, b)$ 에서 $x$ 는 상수, $W, b$ 는 변수. 미분
  대상의 구분이 정적 타입에 그대로 옮겨짐.
- `Constant.grad`, `Constant.backward()` 같은 잘못된 사용이 컴파일
  시점에 잡힘. 런타임 None 체크 / AttributeError 우회 안 됨.
- `Var` 와 `Constant` 의 사용 코드가 의미를 정확히 전달:
  ```python
  x = Constant(x_batch)         # "이건 미분 대상 아니에요"
  W = Var(weight_init)          # "이건 미분 대상이에요"
  ```
- `requires_grad=True` 인자가 사라짐. 사용자가 매번 플래그 안 써도 됨.

**부담**:
- functional 함수가 overload 3 개씩 (`Var`/`Constant` 분기). 일회성 비용,
  사용자한텐 안 보임.
- `Op.apply` 가 입력 보고 결과 클래스 dispatch. `any(isinstance(n, Var)
  for n in inputs)` 한 줄.

### 6.3. Parameter 통합 — 별도 클래스 안 만듦

**결정**: `Parameter` 같은 별도 클래스 없음. 학습 leaf 인지는 `Var`
이고 `_op is None` 인지로 판단, `is_parameter` property.

**이전 결정 변천**: 처음엔 (이전 버전 §1.4) "Parameter 통합" 으로
두면서도 단일 클래스 (`Node`) 안에 모든 메타를 두려 했음. 그러다 §6.2
의 `Constant` / `Var` 분리 결정에서, "추적 vs 비추적" 한 축은 클래스로
분리하되 "학습 leaf vs 중간" 한 축은 `_op` 메타로 두기로 정함. 즉
2-계층 (Constant, Var) + property 1 개 (`is_parameter`).

**고려한 대안**: 3-계층 (`Var` → `TrackedVar` → `Parameter`).
**기각 이유**:
- axon 에 "추적되지만 학습 안 하는" 케이스가 거의 없음. BatchNorm 의
  running_mean 같은 건 `Var(..., optimizer=None)` 또는 그냥 `Constant`
  로 충분.
- `Parameter` 가 `TrackedVar` 의 자식이 되면 `Op.apply` 가 결과로
  `TrackedVar` (학습 leaf 아님) 만들 때, `Parameter` 와 구분하기
  어색해짐.
- 클래스 수 늘리는 비용 대비 얻는 게 적음. property 한 줄로 충분.

### 6.4. UnaryOp / BinaryOp 통합

**결정**: `Op` 단일 base. arity 별 분리 없음.

**대안**: 처음엔 `UnaryOp(Op)`, `BinaryOp(Op)` 로 나눴음.

**이유**:
- 분리의 가치는 base 가 공통 로직을 가질 때 발생. 지금은 인자 unpack
  한 줄, tuple wrap 한 줄이 전부 — 무게 안 나감.
- `Where` (ternary), `Concat` (variadic) 가 곧 도입. 분리 유지하면
  `TernaryOp`, `VariadicOp` 끝없이 늘거나 일부만 `Op` 직접 상속해서
  일관성 깨짐.
- 자식 Op 첫 줄의 `(x,) = inputs` / `a, b = inputs` 가 인자 개수를 코드
  첫 줄에 명시 — 가독성에 오히려 도움.

### 6.5. 연산자 오버로딩 v1 까지 미룸

**결정**: v0 에서는 명시 함수 (`F.add(x, y)`) 만. 연산자 오버로딩은
`_coerce_pair` 헬퍼와 함께 v1 에 도입.

**대안**: v0 부터 dunder 로 `x + y` 지원.

**이유**:
- v0 의 목표는 backward 정확성 검증. 명시 호출이 디버깅 용이.
- backend Array 와 그래프 노드의 dispatch 우선순위 매커니즘이 백엔드마다
  달라 (`__array_priority__` 는 NumPy 만), v0 시점에 dunder 도입하면
  silent 버그 가능성.
- v1 에서 `_coerce_pair` 한 곳에 lifting 매커니즘 모이면 dunder 가
  깔끔한 1:1 위임이 됨.

### 6.6. Pow 패밀리 dispatch

**결정**: `pow` 가 인자 타입 보고 `Pow` / `PowConstExp` / `PowConstBase`
중 하나로 dispatch.

**대안**: 단일 `Pow` Op 만 두고 backward 에서 항상 `xp.log(a)` 호출.

**이유**:
- `Pow.backward` 의 $g \cdot a^b \cdot \log a$ 항이 a < 0 일 때 NaN.
  `x ** 2` 같은 흔한 표현이 NaN 으로 망가짐.
- 지수 / 밑 중 하나가 Scalar 면 그 Op 의 backward 는 log 가 필요 없음.
  별도 Op 으로 빼서 위험 회피.
- 일반적 규칙: 상수 backward 가 위험한 산수 (log/sqrt/div) 부르면 별도
  Op, 안전하면 Scalar lift 로 통합.

### 6.7. Optimizer 가 Var 의 일부 — Net.optimizer default

**결정**: optimizer 가 별도 객체로 떠다니지 않음. 학습 weight 인 `Var`
가 `_optimizer: Optimizer | None` 필드를 가짐. `Net.optimizer` class
attribute 가 default 이고 `__setattr__` 가 weight `Var` 에 자동
deepcopy. 학습 루프는 `loss.backward(); loss.optimize()` 두 줄.

**대안 1 — PyTorch param_groups**:

```python
optimizer = AdamW([
    {'params': layer1.parameters(), 'lr': 1e-5},
    {'params': layer2.parameters(), 'lr': 1e-4},
], lr=1e-4)
```

한 객체, 한 step 호출. 흔한 학습 패턴 깔끔.

단점:
- group 정의가 model 구조에 hard-code. model 바꾸면 group 다시 정의.
- `optimizer.zero_grad()` 매크로 호출 필수 — 누락 시 silent 학습 실패.
- `optimizer.parameters()` 와 model 의 weight 가 따로 노는 동기화 문제.

**대안 2 — Optax multi_transform**:

```python
tx = optax.multi_transform(
    {'backbone': optax.adam(1e-5), 'head': optax.adam(1e-3)},
    param_labels   # pytree, 각 param 에 'backbone' or 'head' 라벨
)
```

매우 유연. 임의의 partition 가능.

단점:
- `param_labels` 가 또 다른 pytree. 관리 복잡도 두 배.
- 함수형 톤이 axon 의 OOP 톤 (Net, forward 메서드) 과 충돌.
- `opt_state` 라는 변수가 학습 루프 밖에 매번 떠다님.

**대안 3 — NNX wrt=type**:

```python
optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
```

Variable 타입 (`nnx.Param`, `nnx.LoRAParam`) 으로 dispatch. 박스 클래스
시스템 기반.

단점:
- axon 은 박스 클래스 (`Parameter`) 를 기각 (§6.3). 이 방식 못 씀.

**axon 방식의 강점** (Var 부착 + Net.optimizer default):

- **State 위치가 정직**: Adam 의 m, v 는 weight 의 history 일부. weight
  옆에 사는 게 자연. weight GC 되면 state 도 자동 GC.
- **Layer-wise LR / weight-decay 분기가 자연스러움**: layer 클래스
  `__init__` 안에서 그냥 다른 optimizer 인스턴스 부착. 별도 group 정의
  매커니즘 없음.
- **Freeze 가 명시적**: `optimizer=None` 또는 `Constant` 로 만들기.
- **학습 루프 두 줄**: `loss.backward(); loss.optimize()`. PyTorch 의
  `zero_grad / backward / step` 세 줄 패턴이 자취 감춤. `step` 안에서
  update 직후 zero.
- **Default 매커니즘으로 일반 케이스도 verbose 안 함**: `Net.optimizer
  = AdamW(lr=...)` 한 줄로 모든 학습 weight 자동 부착.

**약점과 해결**:

- **Global 동작 (gradient clipping, grad norm)**: PyTorch 는
  `clip_grad_norm_(model.parameters())` 같은 함수가 글로벌 norm 처리.
  axon 은 weight 별 optimizer 라 어색. 해결: `Net` 메서드로 둠 —
  `model.clip_grad_norm(1.0)`, `model.grad_norm()`. optimizer 와 직교한
  차원이라 자연.
- **Optimizer state 직렬화**: weight 별로 흩어져 있어서 모으기 어색.
  해결: `model.state_dict()` 가 weight + 그 weight 의 optimizer state 를
  함께 직렬화. PyTorch 처럼 `model.state_dict()` / `optimizer.state_dict()`
  분리 안 함.

**실제 워크로드 검증**: discriminative fine-tuning (BERT 12-layer LR
12 개), backbone vs head, weight-decay 분기 (bias / LayerNorm 제외) 등
흔한 케이스에서 axon 방식이 PyTorch param_groups 보다 자연스러움. layer
클래스 안에서 optimizer 인자로 받으면 끝.

### 6.8. Var / Constant 이름

**결정**: 그래프 노드의 두 클래스는 `Var` / `Constant`.

**고려한 대안**:
- `Node` — 정확하지만 일반적. 너무 광범위.
- `Tensor` — 익숙하지만 그래프와 의미 연결 약함.
- `Expr` — 정확. 학구적이긴 한데 추적 안 되는 leaf 도 같은 타입인데
  "표현식" 으론 어색.
- `Trace` — `requires_grad` 의미와 직결. 영어 "trace" 가 디버깅 / 로깅
  의미와 충돌.
- `TrackedVar` / `Var` — B' 옵션. 어색함 ("왜 추적? 무엇을?").

**이유**:
- 수학 표기 $L = f(x; W, b)$ 에서 $x$ 는 상수 (Constant), $W, b$ 는 변수
  (Var). 미분 가능성과 직결.
- 두 클래스가 정적 타입에서 분기 — 잘못된 사용 (Constant.backward 등) 을
  컴파일 시점에 잡음.
- 3 글자 / 8 글자라 시그니처 가독성 좋음.

### 6.9. Op 비공개

**결정**: Op 클래스는 비공개. 사용자는 `axon.functional` 만 사용.

**대안**: Op 클래스도 export 해서 사용자가 `Add().apply(a, b)` 직접 호출 가능.

**이유**:
- 같은 일을 두 길로 할 수 있으면 어느 게 표준인지 헷갈림. functional
  하나로 일원화.
- Op 시그니처는 내부 추상화 (`validate` 메서드 추가, 인자 순서 조정 등) 라
  바뀔 수 있음. 사용자 노출 안 되면 자유롭게 수정.
- functional 함수가 dispatch 책임 (Pow 패밀리 같은 거) 도 가짐. 사용자가
  Op 직접 호출하면 dispatch 우회.

### 6.10. NamedDim v? 까지 미룸

**결정**: shape 를 dim 이름으로 검증하는 시스템은 NamedDim 박스 클래스
도입할 때까지 보류. v0 ~ v3 는 raw shape (`tuple[int, ...]`) 사용.

**대안**: v0 부터 jaxtyping 같은 매커니즘 도입.

**이유**:
- v0 ~ v2 의 모델은 shape 가 단순 (MNIST: `(B, 784) → (B, 10)`).
  NamedDim 의 가치가 안 나옴.
- Transformer (v3) 부터 `(B, T, C)` / `(B, H, T, T)` 등 차원 늘어나면서
  비로소 효용. 그때 압력 받으면 도입.
- 미리 도입하면 모든 Op 의 validate 가 무거워짐. 검증 비용 vs 디버깅
  편의 trade-off 가 v3 전까진 안 맞음.

---

## 7. 미래 의제

언젠가 들이고 싶지만 지금 압력 안 받는 것들. 도입 시점은 그 압력이
실제로 와야 결정.

### 7.1. NamedDim — shape 의 의미 검증

**무엇**: dim 에 이름 (B, T, C 등) 을 붙여서 shape mismatch 를 의미
수준에서 잡음.

```python
class NamedDim:
    B: int   # batch
    T: int   # time
    C: int   # channel

x: Var[float32, NamedDim("B T C")]
```

**언제**: Transformer 디버깅이 진짜 버거워질 때 (v3 즈음). 그 전엔 raw
shape 검증 (`Op.validate`) 으로 충분.

**구현 후보**:
- jaxtyping 같은 외부 라이브러리 (의존성 추가).
- 자체 NamedDim 시스템 (학습 가치 높음, 구현 비용 큼).

### 7.2. Mixed Precision

**무엇**: forward 는 bfloat16 / float16, master weight 는 float32. 메모리
절반.

**언제**: GPT-2 124M 학습에서 메모리 부족할 때 (v5).

**구현 후보**:
- AMP-style autocast: forward 자동 캐스트, backward 가 master 로 grad 누적.
- 명시적 캐스트: 사용자가 `x.cast(bfloat16)` 직접 호출.

### 7.3. JIT / Compile

**무엇**: Python overhead 를 우회해서 그래프 일부를 GPU 커널로 컴파일.

**언제**: 학습 속도가 진짜 압박일 때 (v5+). MLX 의 `mx.compile` 이 이미
일부 기능 제공해서 axon 레벨에서 추가 매커니즘 필요한지 미정.

### 7.4. Gradient Checkpointing

**무엇**: forward 의 일부 중간 결과를 저장 안 하고 backward 시 재계산.
메모리 절약, 속도 손해.

**언제**: GPT-2 124M from-scratch 학습이 메모리 한계에 부딪힐 때 (v5).

**구현 후보**: `Op` 의 `is_checkpoint` 플래그? 또는 `Net` 의 `checkpoint`
래퍼?

### 7.5. Distributed Training

**무엇**: 여러 GPU / 머신에 model 을 쪼개거나 데이터를 쪼개는 학습.

**언제**: 단일 머신으로 부족할 때. M 시리즈 단일 칩에선 거의 안 와요.
Linux + 여러 GPU 환경 갖추면 그때.

axon 의 backend abstraction 위에 분산 layer 를 얹는 형태가 자연. 미정.

### 7.6. Lazy Tensor / 표현식 융합

**무엇**: forward 가 즉시 계산하지 않고 표현식 트리만 만들어두고 한
번에 융합 (loss 계산 시점에). MLX 가 이미 이걸 하는데 axon 레이어에서
추가로 할 게 있는지.

**언제**: 미정. backend 가 알아서 잘하는지, axon 레벨 추가 매커니즘
필요한지 측정 후 결정.

---

여기까지가 axon 의 설계 메모예요. 두 달 뒤의 저를 위해.
