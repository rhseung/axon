# axon — 설계 철학 및 API 결정

> "왜 이렇게 만들었는가"를 기록한 문서.
> 무엇을 만들지는 [PLAN.md](./PLAN.md) 참고.

---

## 핵심 설계 원칙

### Tensor는 순수한 값이다

`Tensor`는 값(data)과 연산 그래프 정보(`_op`, `_inputs`)만 들고 있다.
grad는 `backward(loss)` 내부의 `grads` dict 안에서만 흐르고, `Parameter`일 때만 바깥으로 누적된다.

```
Tensor    — data + _op + _inputs   (순수한 값, 그래프 노드)
Parameter — Tensor + grad          (옵티마이저가 읽을 grad만 여기에)
```

이 분리로 얻는 것:
- 중간 activation의 grad는 `backward()` 종료 후 자동으로 사라짐 (메모리)
- 옵티마이저는 `Parameter` 목록만 보면 됨 — 그래프 순회 불필요
- `Op.backward`는 grad를 어디에 쓸지 몰라도 됨 — 수학만 반환

### Op.backward는 반환한다, 쓰지 않는다

```python
class Add(Op):
    def backward(self, grad, inputs) -> tuple[NDArray, NDArray]:
        return grad, grad  # (∂/∂a, ∂/∂b) — 어디에 쓸지는 backward()가 결정
```

`backward(loss)`가 반환값을 받아 `Parameter`인 경우에만 누적한다.

### Primitive Op만 역전파를 정의한다

`sigmoid`, `relu` 같은 활성화 함수는 primitive Op의 조합으로 표현한다.
역전파는 자동으로 체인룰을 타고 흐른다.

수치 안정성이 필요한 `LogSoftmax`, `CrossEntropyLoss`만 예외적으로 Op으로 직접 구현한다.

### 세 레이어의 역할 분리

autograd가 동작하려면 반드시 Op 인스턴스를 거쳐야 한다.
`xp`(raw 연산)를 직접 쓰면 그래프가 엮이지 않아 역전파가 불가능하다.

```
axon.functional.exp(x)   ← 사용자 API — Tensor → Tensor, 그래프 엮음
    ↓ Op 인스턴스 호출
Exp()(x)                 ← Op.__call__ — _op, _inputs 연결
    ↓ raw 연산
xp.exp(x._data)          ← backend — Array → Array, 그래프 없음
```

---

## `axon.functional` — 공개 함수 API

`axon.functional`(관례상 `F`로 import)은 사용자가 직접 쓰는 유일한 연산 레이어다.
내부적으로 Op을 통해 그래프를 엮는다.

```python
import axon.functional as F
```

### 산술

```python
# primitive Op 직접 위임
F.add(a, b)          # Add()(a, b)
F.sub(a, b)          # Sub()(a, b)
F.mul(a, b)          # Mul()(a, b)
F.div(a, b)          # Div()(a, b)
F.pow(x, n)          # Pow()(x, n)
F.neg(x)             # Neg()(x)
F.matmul(a, b)       # MatMul()(a, b)
```

연산자 오버로딩(`+`, `-`, `*`, `@` 등)은 내부적으로 이 함수들을 호출한다.

### 수학 함수

```python
# primitive Op 직접 위임
F.exp(x)             # Exp()(x)
F.log(x)             # Log()(x)
F.sqrt(x)            # Sqrt()(x)
F.abs(x)             # Abs()(x)
F.sin(x)             # Sin()(x)
F.cos(x)             # Cos()(x)
```

### 활성화 함수 — primitive 조합

별도 Op 없이 primitive를 조합해 자동으로 역전파가 흐른다.

```python
def sigmoid(x: Tensor) -> Tensor:
    return (F.exp(-x) + 1) ** -1

def tanh(x: Tensor) -> Tensor:
    e2x = F.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def relu(x: Tensor) -> Tensor:
    return F.maximum(x, Tensor(xp.zeros_like(x._data)))

def gelu(x: Tensor) -> Tensor:
    # tanh 근사 버전 (GPT-2 스타일)
    return 0.5 * x * (1 + F.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

def silu(x: Tensor) -> Tensor:
    return x * F.sigmoid(x)

def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    return F.maximum(x, negative_slope * x)
```

### 수치 안정성이 필요한 함수 — Op 직접 구현

primitive 조합으로 구현하면 `exp` overflow 등 수치 불안정이 발생하는 경우.

```python
# axon/ops.py 에서 Op으로 직접 구현
class LogSoftmax(Op):
    def forward(self, x):
        shifted = x._data - xp.max(x._data, axis=-1, keepdims=True)  # overflow 방지
        log_sum_exp = xp.log(xp.sum(xp.exp(shifted), axis=-1, keepdims=True))
        return Tensor(shifted - log_sum_exp)

    def backward(self, grad, x):
        softmax = xp.exp(self.forward(x)._data)
        return (Tensor(grad._data - xp.sum(grad._data, axis=-1, keepdims=True) * softmax),)

# functional.py에서 노출
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return LogSoftmax(axis=axis)(x)

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return F.exp(F.log_softmax(x, axis=axis))
```

### 형상 변환

```python
def reshape(x: Tensor, shape: tuple) -> Tensor:
    return Reshape(shape)(x)

def transpose(x: Tensor, axes: tuple | None = None) -> Tensor:
    return Transpose(axes)(x)

def unsqueeze(x: Tensor, axis: int) -> Tensor:
    return Reshape((..., 1))(x)           # expand_dims Op 위임

def squeeze(x: Tensor, axis: int | None = None) -> Tensor:
    return Squeeze(axis)(x)

def flatten(x: Tensor, start_dim: int = 0) -> Tensor:
    new_shape = x.shape[:start_dim] + (-1,)
    return F.reshape(x, new_shape)
```

### 축소

```python
def sum(x: Tensor, axis=None, keepdims=False) -> Tensor:
    return Sum(axis=axis, keepdims=keepdims)(x)

def mean(x: Tensor, axis=None, keepdims=False) -> Tensor:
    return Mean(axis=axis, keepdims=keepdims)(x)

def max(x: Tensor, axis=None, keepdims=False) -> Tensor:
    return Max(axis=axis, keepdims=keepdims)(x)
```

### 인덱싱

```python
def gather(x: Tensor, indices: Tensor, axis: int) -> Tensor:
    return Gather(axis=axis)(x, indices)

def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return Where()(condition, x, y)
```

### PyTorch `F`와의 차이

| | `torch.nn.functional` | `axon.functional` |
|---|---|---|
| 상태 | 없음 | 없음 |
| 역전파 | Tensor.grad 누적 | grads dict 내부 흐름 |
| 활성화 | Op 직접 | primitive 조합 (LogSoftmax 등 예외) |
| 네이밍 | `F.relu`, `F.softmax` | 동일 |

`axon.functional`과 `torch.nn.functional`은 거의 동일한 네이밍을 따른다.
PyTorch 코드를 axon으로 포팅할 때 `import torch.nn.functional as F` → `import axon.functional as F`만 바꾸면 대부분 동작한다.

사용자는 `axon.functional`만 쓴다.
`xp`를 직접 호출하면 autograd가 동작하지 않는다.

```python
import axon.functional as F

y = F.exp(x)       # ✓ 그래프 엮임, backward 동작
y = xp.exp(x)      # ✗ raw 연산, backward 불가
```

---

## PyTorch와의 차별화

### 1. `backward(loss)` — zero_grad 자동화

**문제:** PyTorch에서 `zero_grad()`를 까먹으면 grad가 누적되어 조용한 버그가 발생한다.

```python
# PyTorch — 순서 틀리거나 zero_grad 누락 시 버그
optimizer.zero_grad()
loss.backward()
optimizer.step()

# axon — zero_grad 불필요, 의도가 명확할 때만 accumulate
backward(loss)                        # 매번 grad 초기화
backward(loss, accumulate=True)       # gradient accumulation 시 명시
optimizer.step()
```

**설계:** `backward(loss)`는 시작 시점에 그래프 내 모든 `Parameter.grad`를 초기화한다.
gradient accumulation이 필요한 경우만 `accumulate=True`로 명시적으로 선언한다.

---

### 2. `eval_mode()` — 컨텍스트 매니저

**문제:** PyTorch에서 `model.eval()` 후 `model.train()` 복귀를 잊으면 Dropout, BatchNorm이 잘못 동작한다.

```python
# PyTorch — 상태 관리를 직접 해야 함
model.eval()
with torch.no_grad():
    result = model(x)
model.train()  # 까먹으면 버그

# axon — 블록 종료 시 자동 복귀 + no_grad 동시 적용
with model.eval_mode():
    result = model(x)
```

**설계:** `eval_mode()`는 `no_grad` 컨텍스트를 내장하고, 블록 종료 시 자동으로 train 모드로 복귀한다.

---

### 3. `Tensor.assert_shape(*shape)` — 명시적 shape 검증

**문제:** PyTorch에서 shape 에러는 실제 연산 시점에 뒤늦게 터진다.

```python
# axon — 의도한 shape를 forward 안에서 명시
def forward(self, x):
    x.assert_shape(-1, 784)    # -1은 임의의 배치 크기
    x = self.linear1(x)
    x.assert_shape(-1, 128)    # 여기서 틀리면 즉시 에러
    return x
```

**설계:** `-1`은 임의의 크기를 허용한다. 불일치 시 shape와 기대값을 함께 출력한다.

---

### 4. `Net.__repr__` — tree + 파라미터 수

**문제:** PyTorch의 `__repr__`은 구조는 보여주지만 파라미터 수가 없다.

```python
print(model)
# Sequential
# ├── Linear(784 → 256)    200,960
# ├── ReLU                       0
# ├── Linear(256 → 64)      16,448
# ├── ReLU                       0
# └── Linear(64 → 10)          650
#
# Total parameters: 218,058
```

---

### 5. `Net.summary(input_shape)` — 레이어별 output shape 포함

**문제:** PyTorch는 `torchinfo` 같은 외부 라이브러리가 필요하다.

```python
model.summary(input_shape=(784,))
# Layer              Output Shape    Params
# ─────────────────────────────────────────
# Linear(784 → 256)  (N, 256)       200,960
# ReLU               (N, 256)             0
# Linear(256 → 10)   (N, 10)          2,570
# ─────────────────────────────────────────
# Total params: 203,530
```

---

### 6. `Tensor.item()` — 타입 추론

**문제:** PyTorch의 `.item()`은 반환 타입이 `float | int | bool`로 넓어서 타입 체커가 잘 못 잡는다.

```python
loss = cross_entropy(logits, t)   # Tensor[float32], shape=()
val: float = loss.item()          # pyrefly가 float임을 정확히 추론
```

**설계:** scalar `Tensor`의 dtype에서 Python 타입을 추론한다. `float32/float64 → float`, `int32/int64 → int`, `bool → bool`.

---

### 7. `DataLoader` — Tensor를 직접 yield

**문제:** PyTorch는 DataLoader가 반환하는 값을 매번 `torch.tensor()`로 변환해야 한다.

```python
# PyTorch
for x, t in dataloader:
    x = x.float()    # 또는 torch.tensor(x)

# axon — 처음부터 Tensor로 yield
for x, t in DataLoader(dataset, batch_size=64):
    logits = model(x)    # 바로 사용 가능
```

---

### 8. `checkpoint` — 사람이 읽을 수 있는 포맷

**문제:** PyTorch의 `.pt`는 pickle이라 파일을 열어볼 수 없다.

```
# axon checkpoint (디렉토리 구조)
run/epoch_10.axon/
├── meta.json          ← epoch, loss, timestamp, model 구조
├── linear1.W.npy
├── linear1.b.npy
└── linear2.W.npy
```

numpy로 직접 로드하거나 확인할 수 있어 GPT-2 가중치 로드 검증 때도 유용하다.

---

### 9. `Parameter` 타입 힌트 강화

**문제:** PyTorch의 `Parameter`는 사실상 `requires_grad=True`인 `Tensor`에 마커만 붙인 것이라, 어떤 파라미터인지 타입 수준에서 알 수 없다.

```python
# PyTorch — parameters()가 뭘 반환하는지 타입 정보 없음
model.parameters()  # → Iterator[Parameter], 다 똑같이 생김

# axon — 필드 선언에서 shape/dtype 의도를 명시
class Linear(Net):
    W: Parameter   # shape=(out_features, in_features)
    b: Parameter   # shape=(out_features,)

    def __init__(self, in_features: int, out_features: int):
        self.W = Parameter(kaiming_uniform(out_features, in_features))
        self.b = Parameter(np.zeros(out_features))
```

**설계:** `Parameter`를 클래스 어트리뷰트로 타입 힌트하면 `pyrefly`가 정적 분석으로 잡아준다.
`Net.parameters()`는 `__dict__`를 재귀 순회해 `Parameter` 인스턴스를 자동 수집하므로,
필드 선언이 곧 등록이 된다 — 별도 `register_parameter()` 불필요.

---

### 10. `check_gradients` — numerical gradient check 내장

**문제:** PyTorch의 `torch.autograd.gradcheck`은 사용법이 불편하다.

```python
from axon.testing import check_gradients

check_gradients(model, x)
# ✓ linear1.W  max_diff=1.2e-6  (OK)
# ✓ linear1.b  max_diff=8.3e-7  (OK)
# ✗ linear2.W  max_diff=3.1e-3  (FAIL — 수치 불안정 의심)
```

각 레이어/연산 구현 직후 바로 돌릴 수 있도록 first-class 도구로 제공한다.

---

### 11. `BinaryOp` / `UnaryOp` 분리 — 반환 타입 보장

**문제:** `Op.backward`가 `tuple[Tensor, ...]`를 반환하면 inputs 개수와 반환 개수가 맞는지 런타임에서만 확인된다.

```python
# 지금 — *inputs 언패킹, 반환 개수 보장 없음
class Add(Op):
    def backward(self, grad, *inputs) -> tuple[Tensor, ...]: ...

# 강화 — 입력/반환 개수가 타입 수준에서 보장됨
class UnaryOp[D: DTypeLike](Op[D]):
    @abstractmethod
    def forward(self, x: Tensor[D]) -> Tensor[D]: ...

    @abstractmethod
    def backward(self, grad: Tensor[D], x: Tensor[D]) -> tuple[Tensor[D]]: ...

class BinaryOp[D: DTypeLike](Op[D]):
    @abstractmethod
    def forward(self, a: Tensor[D], b: Tensor[D]) -> Tensor[D]: ...

    @abstractmethod
    def backward(self, grad: Tensor[D], a: Tensor[D], b: Tensor[D]) -> tuple[Tensor[D], Tensor[D]]: ...

# 구현이 명확해짐
class Exp(UnaryOp):
    def forward(self, x): return Tensor(np.exp(x._data))
    def backward(self, grad, x) -> tuple[Tensor]:
        return (Tensor(grad._data * np.exp(x._data)),)

class Add(BinaryOp):
    def forward(self, a, b): return Tensor(a._data + b._data)
    def backward(self, grad, a, b) -> tuple[Tensor, Tensor]:
        return (grad, grad)
```

**설계:** `*inputs` 언패킹 없이 이름 있는 인자로 받아 실수를 줄이고 가독성을 높인다.
`pyrefly`가 반환 tuple 길이를 정적으로 검증한다.

---

### 12. `@` 연산자 — Sequential 체이닝

**문제:** 레이어가 많을 때 `Sequential([...])` 리스트가 장황해진다.

```python
# 지금
model = Sequential([Linear(784, 128), ReLU(), Linear(128, 10)])

# @ 체이닝 — 레이어 합성을 연산자로 표현
model = Linear(784, 128) @ ReLU() @ Linear(128, 10)
```

**설계:** `Net.__matmul__`이 `Sequential`을 반환한다. 이미 `Sequential`이면 중첩 없이 펼쳐서 합친다.

```python
class Net:
    def __matmul__(self, other: Net) -> Sequential:
        lhs = self.modules if isinstance(self, Sequential) else [self]
        rhs = other.modules if isinstance(other, Sequential) else [other]
        return Sequential(lhs + rhs)
```

`A @ B @ C`는 `Sequential([A, B, C])`와 동일하고 타입도 `Sequential`로 추론된다.

---

## 학습 루프 표준 패턴

```python
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
])

optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = CrossEntropyLoss()

for x, t in DataLoader(train_dataset, batch_size=64):
    logits = model(x)          # forward — 순수한 값의 흐름
    loss = loss_fn(logits, t)

    backward(loss)             # zero_grad 자동, grad는 Parameter에만 누적
    optimizer.step()           # Parameter.data -= lr * Parameter.grad

# 평가
with model.eval_mode():
    for x, t in DataLoader(val_dataset, batch_size=256):
        logits = model(x)
        ...
```

---

## 구현 우선순위

실용적인 구현 순서:

```
BinaryOp/UnaryOp  → Op 설계 중인 지금이 마지막 기회
functional.py     → Op 구현과 동시에, 사용자 API 레이어 확립
Parameter 타입    → 지금 당장, pyrefly + 필드 선언이 곧 등록
assert_shape      → 개발 중 디버깅이 바로 편해짐
@ 체이닝          → Sequential 완성 직후, Net.__matmul__ 한 줄
__repr__ tree     → 모델 구조 파악에 즉시 효과
item() 타입       → pyrefly 있으니 바로 활용 가능
eval_mode()       → Dropout 구현 전에 필수
check_gradients   → 각 Op 구현 후 즉시 검증
DataLoader Tensor → 학습 루프 완성 시 필요
summary()         → 있으면 편하지만 급하지 않음
checkpoint        → 학습이 길어지면 바로 필요
```
