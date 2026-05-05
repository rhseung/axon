# axon

> 자동 미분 엔진 + 신경망 프레임워크. **GPT-2 학습/추론까지.**

이 문서는 axon 프로젝트의 단일 reference — 설계 철학, API 결정, 구현 계획, 마일스톤,
파일 구조, 학습 자료까지 모두 여기 모인다. (이전 PLAN.md / DESIGN.md 통합본.)

---

## 목표

- Autograd 직접 구현 (computation graph + chain rule)
- PyTorch / Flax NNX 에서 좋은 부분만 가져온 API
- 백엔드 추상화: **NumPy** (CPU) / **MLX** (Apple Silicon Metal) / **CuPy** (NVIDIA, 옵션)
- 검증 단계별 마일스톤
  - MLP — MNIST val 97%+
  - CNN — CIFAR-10 (선택)
  - RNN/LSTM — TinyShakespeare char-level
  - **Transformer — TinyStories or TinyShakespeare BPE 학습**
  - **GPT-2 124M 가중치 로드 → inference 결과가 HuggingFace 와 동일**

---

## 설계 철학

### Array가 순수한 값이다 (Tensor 클래스를 두지 않은 이유)

backend native ndarray (`numpy.ndarray` / `mlx.array` / `cupy.ndarray`) 가
"순수한 값" 의 역할 — shape/dtype, 연산자, `Self` 반환 — 을 이미 다 가진다.
별도 `Tensor` wrapper 를 두면 단순 통과 layer 가 되어 코드만 두꺼워진다.

대신 `Node` 가 backend `Array` 를 직접 wrap 하면서 그래프 위치 정보 (`_op`,
`_inputs`, `_requires_grad`) 를 얹는다. `Parameter` 는 `Node` 를 상속해 영구
grad 버퍼를 추가한다.

```
Array     — backend native ndarray                          (순수한 값)
Node      — Array + _op + _inputs + _requires_grad          (그래프 노드)
Parameter — Node + grad                                     (학습 가능 leaf)
```

이 분해로 얻는 것:

- Op.forward / backward 가 backend Array 만 다룸 → numpy 사용자가 읽을 수 있는
  수식 코드 그대로 (`return a + b`)
- 중간 activation 의 grad 는 `backward()` 종료 후 자동으로 사라짐 (메모리)
- 옵티마이저는 `Parameter` 목록만 보면 됨 — 그래프 순회 불필요
- `Op.backward` 는 grad 를 어디에 쓸지 몰라도 됨 — 수학만 반환

### `_requires_grad` 는 graph 추적 여부, Parameter 는 학습 leaf — 둘은 다른 차원

| | tracked (`_requires_grad`) | trainable leaf (`Parameter`) |
|---|---|---|
| 입력 데이터 `x` | ✗ | ✗ |
| 중간 결과 `x @ W` | ✓ | ✗ |
| 가중치 `W` | ✓ | ✓ |

`x @ W` 는 학습 leaf 가 아니지만 backward 그래프에는 포함되어야 한다 — 즉 둘은
일치하지 않고 후자가 더 좁다. `Op.apply` 가 입력 중 하나라도 `_requires_grad` 면
출력의 `_requires_grad` 를 True 로 자동 전파해서 이 차이를 표현한다.

`isinstance(t, Parameter)` 로 graph 추적을 대체하면 chain rule 이 끊기므로 안 된다.
`Parameter` 는 별도로 `Net.parameters()` 가 학습 대상을 식별하는 마커 + `.grad`
영구 버퍼 소유자 역할만 한다.

### Op.backward는 반환한다, 쓰지 않는다

```python
class Add(BinaryOp):
    def backward_binary(self, grad, a, b) -> tuple[Array, Array]:
        return grad, grad  # (∂/∂a, ∂/∂b) — 어디에 쓸지는 backward()가 결정
```

grad 를 `inputs[i].grad` 에 직접 쓰지 않는다. `backward(loss)` 가 반환값을 받아
`Parameter` 인 경우에만 누적한다.

### Op.forward / backward 는 Array, Op.apply 는 Node

`forward` / `backward` 가 Node 를 받으면 `n._data` 같은 unwrap 노이즈가 시그니처
안에 침투한다. 그래서 시그니처를 Array 로 좁히고, Node ↔ Array 변환은 `Op.apply`
한 곳에서 처리한다.

```python
class Op:
  def apply(self, *inputs: Node[D]) -> Node[D]:
    out_array = self.forward(*(n._data for n in inputs))   # Node → Array
    out = Node.from_array(out_array)
    if any(n._requires_grad for n in inputs):
      out._op, out._inputs, out._requires_grad = self, inputs, True
    return out
```

이 분리로 `Op.forward` 본문이 `return a + b` 처럼 진짜 numpy 코드가 된다.

### Primitive Op만 역전파를 정의한다

`sigmoid`, `relu` 같은 활성화 함수는 primitive Op 의 조합으로 표현한다.
역전파는 자동으로 체인룰을 타고 흐른다.

수치 안정성이 필요한 `LogSoftmax`, `CrossEntropyLoss` 만 예외적으로 Op 으로
직접 구현한다.

### 상수는 Op 인스턴스 필드로

`pow(x, 2)` 의 `2`, `sum(x, axis=1)` 의 `1`, `reshape(x, (3, 4))` 의 `(3, 4)` —
사용자가 명시한 상수는 `_inputs` 가 아니라 Op 인스턴스 필드로 보관한다.

```python
class PowConstExp(UnaryOp):       # _inputs = (x,) — n 은 필드
  def __init__(self, n: Scalar):
    self.n = n
  def forward_unary(self, x):
    return x ** self.n
  def backward_unary(self, grad, x):
    return grad * self.n * x ** (self.n - 1)   # log 항 없음 → 음수 base 안전
```

`functional.pow` 가 입력 타입을 보고 `Pow` / `PowConstExp` / `PowConstBase` 로
dispatch:
- 둘 다 Node → `Pow` (backward 에 log 항 — a > 0 강제)
- 지수 상수 → `PowConstExp(n)` (n 은 Op 필드)
- 밑 상수 → `PowConstBase(c)` (c 는 Op 필드)

이 패턴은 axis (`sum(x, axis=1)`), shape (`reshape(x, (3, 4))`) 등에도 일반화된다.
**일반화 규약: `_inputs` 는 Node 만, 그 외 (Scalar / shape / axis 등) 는 Op 인스턴스
필드.**

### 세 레이어의 역할 분리

autograd 가 동작하려면 반드시 Op.apply 를 거쳐야 한다.
`xp` (raw 연산) 를 직접 쓰면 그래프가 엮이지 않아 역전파가 불가능하다.

```
axon.functional.exp(x)   ← 사용자 API — Node → Node, 그래프 엮음
    ↓ Op 인스턴스화 + apply
Exp().apply(x)           ← Op.apply — _op, _inputs 연결, _requires_grad 전파
    ↓ Node unwrap 후 raw 연산
xp.exp(x._data)          ← backend — Array → Array, 그래프 없음
```

---

## `axon.functional` — 공개 함수 API

`axon.functional` (관례상 `F` 로 import) 은 사용자가 직접 쓰는 유일한 연산 레이어다.
내부적으로 Op 을 통해 그래프를 엮는다.

```python
import axon.functional as F
```

### 산술

```python
# primitive Op 직접 위임
F.add(a, b)          # Add().apply(a, b)
F.sub(a, b)          # add(a, neg(b))    — 합성, primitive 아님
F.mul(a, b)          # Mul().apply(a, b)
F.div(a, b)          # Div().apply(a, b)
F.pow(x, n)          # 입력 타입 dispatch (Pow / PowConstExp / PowConstBase)
F.neg(x)             # Neg().apply(x)
F.matmul(a, b)       # MatMul().apply(a, b)
```

연산자 오버로딩 (`+`, `-`, `*`, `@`, `**` 등) 은 내부적으로 이 함수들을 호출한다.
`x ** 2` → `F.pow(x, 2)` → `PowConstExp(2).apply(x)` 로 풀린다.

### 수학 함수

```python
# primitive Op 직접 위임
F.exp(x)             # Exp().apply(x)
F.log(x)             # Log().apply(x)
F.sqrt(x)            # Sqrt().apply(x)
F.abs(x)             # Abs().apply(x)
F.sin(x)             # Sin().apply(x)
F.cos(x)             # Cos().apply(x)
```

### 활성화 함수 — primitive 조합

별도 Op 없이 primitive 를 조합해 자동으로 역전파가 흐른다.

```python
def sigmoid(x: Node) -> Node:
    return (F.exp(-x) + 1) ** -1

def tanh(x: Node) -> Node:
    e2x = F.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def relu(x: Node) -> Node:
    return F.maximum(x, Node(xp.zeros_like(x._data)))

def gelu(x: Node) -> Node:
    # tanh 근사 버전 (GPT-2 스타일)
    return 0.5 * x * (1 + F.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

def silu(x: Node) -> Node:
    return x * F.sigmoid(x)

def leaky_relu(x: Node, negative_slope: float = 0.01) -> Node:
    return F.maximum(x, negative_slope * x)
```

### 수치 안정성이 필요한 함수 — Op 직접 구현

primitive 조합으로 구현하면 `exp` overflow 등 수치 불안정이 발생하는 경우.
이런 Op 은 `forward` / `backward` 시그니처에 맞춰 Array 만 다룬다.

```python
# axon/operation/log_softmax.py
class LogSoftmax(UnaryOp):
    def __init__(self, axis: int = -1):
        self.axis = axis     # 상수는 Op 인스턴스 필드

    def forward_unary(self, x: Array) -> Array:
        shifted = x - xp.max(x, axis=self.axis, keepdims=True)   # overflow 방지
        return shifted - xp.log(xp.sum(xp.exp(shifted), axis=self.axis, keepdims=True))

    def backward_unary(self, grad: Array, x: Array) -> Array:
        softmax = xp.exp(self.forward_unary(x))
        return grad - xp.sum(grad, axis=self.axis, keepdims=True) * softmax

# functional.py 에서 노출
def log_softmax(x: Node, axis: int = -1) -> Node:
    return LogSoftmax(axis=axis).apply(x)

def softmax(x: Node, axis: int = -1) -> Node:
    return F.exp(F.log_softmax(x, axis=axis))
```

### 형상 변환

```python
def reshape(x: Node, shape: tuple) -> Node:
    return Reshape(shape).apply(x)              # shape 은 Op 필드

def transpose(x: Node, axes: tuple | None = None) -> Node:
    return Transpose(axes).apply(x)

def unsqueeze(x: Node, axis: int) -> Node:
    return Unsqueeze(axis).apply(x)

def squeeze(x: Node, axis: int | None = None) -> Node:
    return Squeeze(axis).apply(x)

def flatten(x: Node, start_dim: int = 0) -> Node:
    new_shape = x.shape[:start_dim] + (-1,)
    return F.reshape(x, new_shape)
```

### 축소

```python
def sum(x: Node, axis=None, keepdims=False) -> Node:
    return Sum(axis=axis, keepdims=keepdims).apply(x)

def mean(x: Node, axis=None, keepdims=False) -> Node:
    return Mean(axis=axis, keepdims=keepdims).apply(x)

def max(x: Node, axis=None, keepdims=False) -> Node:
    return Max(axis=axis, keepdims=keepdims).apply(x)
```

### 인덱싱

```python
def gather(x: Node, indices: Node, axis: int) -> Node:
    return Gather(axis=axis).apply(x, indices)

def where(condition: Node, x: Node, y: Node) -> Node:
    return Where().apply(condition, x, y)
```

### PyTorch `F` 와의 차이

| | `torch.nn.functional` | `axon.functional` |
|---|---|---|
| 상태 | 없음 | 없음 |
| 역전파 | Tensor.grad 누적 | Parameter.grad 만 누적 (중간은 dict) |
| 활성화 | Op 직접 | primitive 조합 (LogSoftmax 등 예외) |
| 상수 인자 | Tensor wrap 필요 (`torch.tensor(2)`) | Python scalar 직접 (`F.pow(x, 2)`) — Op 필드 dispatch |
| 네이밍 | `F.relu`, `F.softmax` | 동일 |

`axon.functional` 과 `torch.nn.functional` 은 거의 동일한 네이밍을 따른다.
PyTorch 코드를 axon 으로 포팅할 때 `import torch.nn.functional as F` →
`import axon.functional as F` 만 바꾸면 대부분 동작한다.

사용자는 `axon.functional` 만 쓴다.
`xp` 를 직접 호출하면 autograd 가 동작하지 않는다.

```python
import axon.functional as F

y = F.exp(x)         # ✓ Node → Node, 그래프 엮임, backward 동작
y = xp.exp(x._data)  # ✗ Array → Array, 그래프 없음
```

---

## PyTorch 와의 차별화

### 1. `backward(loss)` — zero_grad 자동화

**문제:** PyTorch 에서 `zero_grad()` 를 까먹으면 grad 가 누적되어 조용한 버그가 발생한다.

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

**설계:** `backward(loss)` 는 시작 시점에 그래프 내 모든 `Parameter.grad` 를 초기화한다.
gradient accumulation 이 필요한 경우만 `accumulate=True` 로 명시적으로 선언한다.

---

### 2. `eval_mode()` — 컨텍스트 매니저

**문제:** PyTorch 에서 `model.eval()` 후 `model.train()` 복귀를 잊으면 Dropout, BatchNorm 이 잘못 동작한다.

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

**설계:** `eval_mode()` 는 `no_grad` 컨텍스트를 내장하고, 블록 종료 시 자동으로 train 모드로 복귀한다.

---

### 3. `Node.assert_shape(*shape)` + Named Dimensions — 명시적 shape 검증

**문제:** PyTorch 에서 shape 에러는 실제 연산 시점에 뒤늦게 터진다. 또한 `x.shape[0]`, `256` 같은 숫자만 코드에 있어서 의미가 흐려진다.

```python
# 기본 — 정수 / None 혼용
def forward(self, x):
    x.assert_shape(-1, 784)    # -1 은 임의의 배치 크기
    x = self.linear1(x)
    x.assert_shape(-1, 128)    # 여기서 틀리면 즉시 에러
    return x

# Named Dim — forward 안에서 shape 를 이름으로 bind
def forward(self, x):
    B, T, C = x.dims("B T C")              # shape 에서 즉시 bind
    H, D = self.H, C // H                  # DimExpr — 산술도 가능
    q = self.Wq(x).assert_shape(B, T, C).reshape(B, T, H, D)
    # ShapeError: dim C expected 256, got 512
    #   assertion at  attention.py:47   q.assert_shape(B, T, C)
    #   C was bound at attention.py:34  B, T, C = x.dims("B T C")
```

**설계:** 자세한 API / 에러 메시지 / 구현 단계는 [NAMED_DIMS.md](./NAMED_DIMS.md). 정수만 쓰는 단순 `assert_shape` 는 Named Dim 구현의 부분집합으로 자연스럽게 흡수.

---

### 4. `Net.__repr__` — tree + 파라미터 수

**문제:** PyTorch 의 `__repr__` 은 구조는 보여주지만 파라미터 수가 없다.

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

**문제:** PyTorch 는 `torchinfo` 같은 외부 라이브러리가 필요하다.

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

### 6. `Node.item()` — 타입 추론

**문제:** PyTorch 의 `.item()` 은 반환 타입이 `float | int | bool` 로 넓어서 타입 체커가 잘 못 잡는다.

```python
loss = cross_entropy(logits, t)   # Node[float32], shape=()
val: float = loss.item()          # pyrefly 가 float 임을 정확히 추론
```

**설계:** scalar `Node` 의 dtype 에서 Python 타입을 추론한다. `float32/float64 → float`, `int32/int64 → int`, `bool → bool`.

---

### 7. `DataLoader` — Node 를 직접 yield

**문제:** PyTorch 는 DataLoader 가 반환하는 값을 매번 `torch.tensor()` 로 변환해야 한다.

```python
# PyTorch
for x, t in dataloader:
    x = x.float()    # 또는 torch.tensor(x)

# axon — 처음부터 Node 로 yield
for x, t in DataLoader(dataset, batch_size=64):
    logits = model(x)    # 바로 사용 가능
```

---

### 8. `checkpoint` — 사람이 읽을 수 있는 포맷

**문제:** PyTorch 의 `.pt` 는 pickle 이라 파일을 열어볼 수 없다.

```
# axon checkpoint (디렉토리 구조)
run/epoch_10.axon/
├── meta.json          ← epoch, loss, timestamp, model 구조
├── linear1.W.npy
├── linear1.b.npy
└── linear2.W.npy
```

numpy 로 직접 로드하거나 확인할 수 있어 GPT-2 가중치 로드 검증 때도 유용하다.

---

### 9. `Parameter` 타입 힌트 강화

**문제:** PyTorch 의 `Parameter` 는 사실상 `requires_grad=True` 인 `Tensor` 에 마커만 붙인 것이라, 어떤 파라미터인지 타입 수준에서 알 수 없다.

```python
# PyTorch — parameters() 가 뭘 반환하는지 타입 정보 없음
model.parameters()  # → Iterator[Parameter], 다 똑같이 생김

# axon — 필드 선언에서 shape/dtype 의도를 명시
class Linear(Net):
    W: Parameter   # shape=(out_features, in_features)
    b: Parameter   # shape=(out_features,)

    def __init__(self, in_features: int, out_features: int):
        self.W = Parameter(kaiming_uniform(out_features, in_features))
        self.b = Parameter(np.zeros(out_features))
```

**설계:** `Parameter` 를 클래스 어트리뷰트로 타입 힌트하면 `pyrefly` 가 정적 분석으로 잡아준다.
`Net.parameters()` 는 `__dict__` 를 재귀 순회해 `Parameter` 인스턴스를 자동 수집하므로,
필드 선언이 곧 등록이 된다 — 별도 `register_parameter()` 불필요.

---

### 10. `check_gradients` — numerical gradient check 내장

**문제:** PyTorch 의 `torch.autograd.gradcheck` 은 사용법이 불편하다.

```python
from axon.testing import check_gradients

check_gradients(model, x)
# ✓ linear1.W  max_diff=1.2e-6  (OK)
# ✓ linear1.b  max_diff=8.3e-7  (OK)
# ✗ linear2.W  max_diff=3.1e-3  (FAIL — 수치 불안정 의심)
```

각 레이어/연산 구현 직후 바로 돌릴 수 있도록 first-class 도구로 제공한다.

---

### 11. `BinaryOp` / `UnaryOp` 분리 — 반환 타입 보장 [구현됨]

`*inputs` 언패킹 없이 이름 있는 인자로 받아 실수를 줄이고 가독성을 높인다.
시그니처는 `Array` 도메인에서 정의 — Op.forward / backward 가 Node 를 모름.

```python
class UnaryOp[D: DType](Op[D]):
    @abstractmethod
    def forward_unary(self, x: Array[D]) -> Array[D]: ...
    @abstractmethod
    def backward_unary(self, grad: Array[D], x: Array[D]) -> Array[D]: ...

class BinaryOp[D: DType](Op[D]):
    @abstractmethod
    def forward_binary(self, a: Array[D], b: Array[D]) -> Array[D]: ...
    @abstractmethod
    def backward_binary(self, grad: Array[D], a: Array[D], b: Array[D]) -> tuple[Array[D], Array[D]]: ...

# 구현이 numpy 코드처럼 깔끔
class Add(BinaryOp):
    def forward_binary(self, a, b): return a + b
    def backward_binary(self, grad, a, b): return (grad, grad)

class PowConstExp(UnaryOp):           # 상수는 인스턴스 필드, _inputs 에 안 들어감
    def __init__(self, n: Scalar): self.n = n
    def forward_unary(self, x): return x ** self.n
    def backward_unary(self, grad, x): return grad * self.n * x ** (self.n - 1)
```

---

### 12. `@` 연산자 — Sequential 체이닝

**문제:** 레이어가 많을 때 `Sequential([...])` 리스트가 장황해진다.

```python
# 지금
model = Sequential([Linear(784, 128), ReLU(), Linear(128, 10)])

# @ 체이닝 — 레이어 합성을 연산자로 표현
model = Linear(784, 128) @ ReLU() @ Linear(128, 10)
```

**설계:** `Net.__matmul__` 이 `Sequential` 을 반환한다. 이미 `Sequential` 이면 중첩 없이 펼쳐서 합친다.

```python
class Net:
    def __matmul__(self, other: Net) -> Sequential:
        lhs = self.modules if isinstance(self, Sequential) else [self]
        rhs = other.modules if isinstance(other, Sequential) else [other]
        return Sequential(lhs + rhs)
```

`A @ B @ C` 는 `Sequential([A, B, C])` 와 동일하고 타입도 `Sequential` 로 추론된다.

---

## API 설계 방향

```python
# 모델 정의 — PyTorch 스타일
class GPT(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)

    def forward(self, idx):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

# 학습
optimizer = AdamW(model.parameters(), lr=3e-4)
backward(loss)
optimizer.step()

# 백엔드 선택
axon.set_backend("mlx")  # or "numpy", "cupy"
```

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
    logits = model(x)          # forward — Node 도메인
    loss = loss_fn(logits, t)

    backward(loss)             # zero_grad 자동, grad 는 Parameter 에만 누적
    optimizer.step()           # Parameter._data -= lr * Parameter.grad

# 평가
with model.eval_mode():
    for x, t in DataLoader(val_dataset, batch_size=256):
        logits = model(x)
        ...
```

---

## 백엔드

### 추상화 레이어

- `axon.backend` 모듈 — 현재 백엔드의 array 라이브러리 노출
- `set_backend("numpy" | "mlx" | "cupy")`
- 공통 함수 시그니처: `array`, `matmul`, `sum`, `reshape`, `where`, `concat`, ...
- `as_numpy(x)` / `from_numpy(x)` — 변환 유틸

### MLX 특이사항 처리

- Lazy evaluation — `mx.eval()` 호출 시점 결정 (forward 끝나고? backward 끝나고?)
- In-place 연산 부재 — `+=` 같은 grad 누적을 어떻게 처리할지
- `mx.float32`, `mx.bfloat16` 등 dtype 매핑
- `mx.compile` 활용 (선택, hot path 최적화)

### 디바이스 관리

- `Node.to(device)` — `"cpu"`, `"gpu"`, `"mps"`
- CPU↔GPU 자동 변환 막기 (성능 함정)

---

## 컴포넌트 / 구현 대상

### Node / Parameter

> **Array 가 순수한 값이다.** Node 는 그래프 노드, Parameter 는 학습 가능 leaf.

#### Node

- `_data: Array` (backend native ndarray), `_op`, `_inputs`, `_requires_grad` 필드
- `dtype`, `shape`, `ndim`, `requires_grad`, `op`, `inputs` property
- `as_numpy()` — numpy 변환
- 연산자 오버로딩 (`__add__`, `__matmul__` 등) → `axon.functional` 로 라우팅
- `detach()` — graph 에서 분리 (선택)
- `to(device)`, `to(dtype)` — backend 단위 (선택)

#### Parameter (Node 상속)

- `grad: Array` 영구 버퍼 (zero_grad 로 초기화)
- `_requires_grad = True` (자동)
- `Net.parameters()` 가 `isinstance(value, Parameter)` 로 수집

### Op

- `Op` 추상 클래스 — `forward(*inputs: Array) -> Array`, `backward(grad, *inputs) -> tuple[Array, ...]`
- `apply(*inputs: Node) -> Node` — Node ↔ Array 변환 + 그래프 metadata 세팅
- `backward` 는 각 input 에 전달할 grad 를 **반환** 한다 — side effect 없음
- `UnaryOp` / `BinaryOp` — `forward_unary` / `forward_binary` 등 이름 있는 인자로 받는 specialization
- 상수는 Op 인스턴스 필드로 보관 (`PowConstExp.n`, `PowConstBase.c` 등)

### 연산 (forward + backward)

#### 산술

- `__add__`, `__sub__`, `__mul__`, `__truediv__`
- `__pow__`, `__neg__`
- `__radd__`, `__rmul__` 등 reverse
- `__matmul__`

#### 축소 / 형상

- `sum`, `mean`, `max`, `min`, `var`, `std`
- `reshape`, `view`, `transpose`, `permute`
- `squeeze`, `unsqueeze`, `expand`
- `concat`, `stack`, `split`, `chunk`

#### 인덱싱

- `__getitem__` — slice / int / fancy indexing / bool mask
- `gather`, `scatter`
- `where` (마스킹용)

#### 수학 함수

- `exp`, `log`, `sqrt`, `rsqrt`
- `abs`, `sign`, `clip`
- `sin`, `cos` (RoPE 에 필요)

### Autograd

- 위상 정렬로 노드 순서 결정
- `backward(loss)` — `grads: dict[id, Array]` 내부에서 grad 흐름
  - `Op.backward` 가 반환한 grad 를 각 input 에 합산
  - `isinstance(inp, Parameter)` 일 때만 `inp.grad +=` 누적
- grad 합산 — 같은 노드가 여러 경로로 쓰일 때 `grads[id]` 에서 자동 누적
- Broadcasting 후 broadcast 된 축 복원
- `keepdims` 처리
- `no_grad()` 컨텍스트 매니저 — `_inputs` 기록 생략
- **Gradient checkpointing** — 메모리 위해 일부 노드 재계산
- **Gradient clipping** — `clip_grad_norm_`

### Activation

- `relu`
- `leaky_relu` (negative_slope 고정값)
- `prelu` — α 가 학습 파라미터, `nn.PReLU` Module + 함수 둘 다
- `elu`
- `gelu` (exact + tanh approx)
- `silu` / `swish`
- `mish`
- `glu` — Gated Linear Unit
- `swiglu` — LLaMA FFN 에서 사용
- `sigmoid`, `tanh`
- `softmax`, `log_softmax` (수치 안정)
- `softplus`

### Layers (`nn`)

#### 베이스

- `Module` — `parameters()`, `zero_grad()`, `train()`, `eval()`
  - `parameters()` — `__dict__` 를 재귀 순회해 `Parameter` 수집
  - `zero_grad()` — `p.grad = zeros_like(p._data)` for p in parameters()
- `Parameter` — `Node` 의 서브클래스. `grad: Array` 필드 추가. 옵티마이저와의 유일한 접점.
- `Sequential`, `ModuleList`, `ModuleDict`
- `state_dict()`, `load_state_dict()` — 가중치 저장/로드 (HF 호환 위해 필수)

#### Linear / Embedding

- `Linear(in, out, bias=True)`
- `Embedding(num_emb, dim)`

#### Convolution

- `Conv1d`, `Conv2d`
- `ConvTranspose2d`

#### Pooling

- `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`

#### Normalization

- `BatchNorm1d`, `BatchNorm2d` — running stats, train/eval
- `LayerNorm`
- `RMSNorm` — LLaMA 계열
- `GroupNorm`
- `InstanceNorm1d`, `InstanceNorm2d` (선택)

#### Regularization

- `Dropout` — train/eval 분기
- `Dropout2d` (CNN 용)

#### Recurrent

- `RNN` cell + 시퀀스 wrapper
- `LSTM`
- `GRU`

#### Residual / Skip

- **Residual connection** — `x + sublayer(x)` 패턴, Transformer/ResNet 공통
- `ResidualBlock` (BasicBlock) — Conv-BN-ReLU-Conv-BN + skip, ResNet-18/34 용
- `Bottleneck` — 1×1 → 3×3 → 1×1 + skip, ResNet-50/101/152 용
- `SqueezeExcitation` (선택) — SE-Net 채널 어텐션

#### Attention

- `ScaledDotProductAttention` — 함수, `softmax(QK^T/√d)V` 수식만
- `SingleHeadAttention` — Module, q/k/v/o projection 포함, head 1개 (학습/디버깅 단계용)
- `MultiHeadAttention` — q/k/v/o projection + head split/merge
- **Causal mask** — `tril`, `-inf` 채우기
- **Padding mask**
- **KV cache** — 추론 시 이전 k/v 재사용
- **Grouped Query Attention** (선택, LLaMA-2 70B)
- **Flash Attention 흉내** — 진짜 fused 는 어렵고, 메모리 효율적인 청크 버전 정도 (선택)

#### Positional Encoding

- **Sinusoidal PE** — 원논문 스타일
- **Learned PE** — GPT-2 스타일
- **RoPE (Rotary Position Embedding)** — LLaMA 스타일
- **ALiBi** (선택)

#### Transformer 블록

- `TransformerEncoderLayer` — Self-Attn + FFN + 2× Norm (Pre-norm)
- `TransformerDecoderLayer` — + Cross-Attn (선택)
- `TransformerBlock` (GPT 스타일, decoder-only)
- FFN 변형: 일반 (Linear-GELU-Linear) / SwiGLU (LLaMA)

#### 기타

- `Flatten`
- `Identity`

### Loss

- `MSELoss`, `L1Loss`, `HuberLoss`
- `CrossEntropyLoss` — stable log_softmax + NLL, ignore_index 옵션
- `NLLLoss`
- `BCELoss`, `BCEWithLogitsLoss`
- **Label smoothing** — CE 옵션
- **Perplexity** — eval 지표 (loss → exp)

### Optimizer

#### 베이스

- `Optimizer(params: list[Parameter])` — `step()`, `zero_grad()`, param group, state dict
- `step()` — `Parameter._data` 를 `Parameter.grad` 로 갱신
- `zero_grad()` — `Parameter.grad = zeros_like`

#### 구현

- `SGD` — momentum, nesterov, weight_decay
- `Adagrad`
- `RMSprop`
- `Adam` — m, v + bias correction
- `AdamW` — decoupled weight decay (Transformer 표준)
- `Lion` (선택)

#### LR Scheduler

- `StepLR`, `MultiStepLR`
- `CosineAnnealingLR` — Transformer 학습에 자주 쓰임
- `LinearWarmup` — 처음 N step 워밍업
- `WarmupCosine` — 위 둘 합성

### 초기화 (`nn.init`)

- `xavier_uniform`, `xavier_normal`
- `kaiming_uniform`, `kaiming_normal`
- `normal_`, `uniform_`
- `zeros_`, `ones_`, `constant_`
- **GPT-2 스타일** — `std=0.02` normal, residual projection 은 `1/sqrt(2*n_layer)` 스케일

### Tokenization

- **Character-level** — 가장 간단, TinyShakespeare 학습용
- **Byte-level BPE (GPT-2 호환)** — `tiktoken` 또는 직접 BPE 학습
- vocab 저장/로드
- HuggingFace tokenizer 호환 (옵션) — GPT-2 가중치 로드 검증용

### Sampling / Generation

- **Greedy decoding**
- **Temperature**
- **Top-k**
- **Top-p (nucleus)**
- **Repetition penalty** (선택)
- **KV cache 활용한 incremental decoding**
- 스트리밍 출력

### 데이터 / 학습 인프라

#### 데이터

- `Dataset` 베이스
- `DataLoader` — 배치, 셔플, num_workers (멀티프로세싱은 선택)
- **MNIST** loader (IDX 파싱)
- **CIFAR-10** loader (선택)
- **TinyShakespeare** loader (텍스트 → 토큰 stream)
- **TinyStories** loader

#### 학습 유틸

- 학습 루프 표준화
- **Mixed precision** — fp32 master + fp16/bf16 forward (MLX 는 bf16 native)
- **Gradient accumulation** — 작은 GPU 에서 큰 batch 흉내
- 체크포인트 저장/재개
- WandB or 간단한 로컬 로거 (loss curve)

---

## HuggingFace 가중치 로드 (검증용)

- GPT-2 (124M) state_dict 다운로드 + 키 매핑
- 같은 입력에 대해 logits 비교 (max abs diff < 1e-4)
- 같은 입력에 대해 generation 결과 비교
- LLaMA 또는 Qwen small (선택, RoPE/RMSNorm/SwiGLU 검증)

---

## 마일스톤별 검증

### 1. MLP / MNIST

- val acc 97%+
- PyTorch 같은 셋업 대비 ±1% 이내

### 2. CNN / CIFAR-10 (선택)

- val acc 70%+ (간단한 모델 기준)

### 2.5. ResNet / CIFAR-10 or ImageNet subset (선택)

- ResNet-18 from scratch
- CIFAR-10 에서 90%+ 또는 Imagenette 등 작은 이미지넷 subset 에서 검증
- BatchNorm + Residual 동작 확인 (둘 다 처음 합쳐 쓸 때 자주 망가지는 부분)

### 3. RNN/LSTM / TinyShakespeare

- 학습 후 그럴듯한 셰익스피어풍 텍스트 생성
- loss 수렴 곡선이 PyTorch 구현과 유사

### 4. Transformer / TinyStories or TinyShakespeare

- decoder-only Transformer 학습
- loss 수렴 (TinyShakespeare 기준 ~1.5)
- 생성 텍스트 그럴듯함

### 5. GPT-2 124M inference

- HF 가중치 로드 후 logits 일치
- Greedy / top-p 생성 결과 일치

### 6. (Stretch) GPT-2 124M 학습

- FineWeb 일부 또는 OpenWebText 일부로 from-scratch 학습 시도
- M 시리즈 한 대로는 시간/메모리 빠듯함, 합리적인 loss 도달이 목표

### 7. (Stretch) Vision Transformer

- ViT-Tiny CIFAR-10 학습 — patch embedding + Transformer encoder
- CNN/Transformer 둘 다 구현된 만큼, 두 패러다임 비교 가능

---

## 테스트

- **Numerical gradient check** — $\frac{f(x+\epsilon) - f(x-\epsilon)}{2\epsilon}$, 오차 < 1e-5
- 모든 연산/activation/layer 커버
- PyTorch 와 forward/backward 결과 비교 (같은 입력 → 같은 출력 / grad)
- 백엔드별 동일 결과 검증 (numpy ↔ mlx)
- toy 문제로 옵티마이저 수렴 확인
- state_dict 저장 후 로드해서 결과 동일

---

## 구현 우선순위

### ✅ 완료

**기반:**
- `backend/` — `Array` Protocol, `BackendProtocol`, `xp` proxy, NumPy / MLX / CuPy 백엔드 + dtype 매핑
- `dtype.py` — `DType` 클래스 계층 (FLOAT16/BF16/FLOAT32/FLOAT64/INT32/INT64/BOOL)
- `scalar.py` — `Scalar = int | float` (functional dispatch + Op 인스턴스 필드용)

**자동미분 코어:**
- `node.py` — `Node` (Array + `_op` + `_inputs` + `_requires_grad`). Tensor 래퍼는 두지 않음.
- `parameter.py` — `Parameter(Node)` 학습 leaf, `.grad: Array` 영구 버퍼
- `_requires_grad` 자동 전파 (`Op.apply` 에서 입력 중 하나라도 True 면 출력 True)
- `backward.py` — 위상 정렬 + `grads: dict[id, Array]` + `Parameter.grad +=` 누적
- property getter (`requires_grad`, `op`, `inputs`)

**Op 시스템:**
- `operation/op.py` — `Op` + `UnaryOp` + `BinaryOp` 베이스. `forward`/`backward` 는 Array 만, `apply` 가 Node ↔ Array 변환
- 산술 primitive: `Add`, `Mul`, `MatMul`, `Div`, `Neg`
- 거듭제곱 + 상수 dispatch: `Pow`, `PowConstExp(n)`, `PowConstBase(c)` — `pow.py` 통합

**functional / 사용자 API:**
- `F.add`, `F.sub`, `F.mul`, `F.div`, `F.neg`, `F.matmul` — primitive Op 위임
- `F.pow` — 입력 타입 dispatch (Pow / PowConstExp / PowConstBase)
- Node 연산자 오버로딩: `+ - * / @ ** neg` 모두 + r-variant + `**` 의 scalar 양방향

**Net:**
- `net.py` — `Net` 베이스 + `parameters()` (재귀 순회로 `Parameter` 수집)

**문서:**
- `CLAUDE.md` 단일 통합 (PLAN/DESIGN 합본)

---

### 🔜 다음 (마일스톤별)

**M0. Op 채우기 (현재 단계 직후)**
- 수학 함수 Op: `Exp`, `Log`, `Sqrt`, `Sin`, `Cos`, `Abs`, `Sign`, `Clip`, `Maximum`, `Minimum`
- 축소 Op (axis/keepdims 필드): `Sum`, `Mean`, `Max`, `Min`, `Var`, `Std`
- 형상 Op (shape/axes 필드): `Reshape`, `Transpose`, `Squeeze`, `Unsqueeze`, `Expand`, `Concat`, `Stack`, `Split`
- 인덱싱: `Gather`, `Where` + `Node.__getitem__` (slice / int / fancy / bool mask)
- 수치 안정 직접 Op: `LogSoftmax`, `CrossEntropyLoss`

**M1. Activation (primitive 조합)**
- `sigmoid`, `tanh`, `relu`, `leaky_relu`, `gelu` (tanh approx + exact), `silu`, `prelu`, `elu`, `mish`, `glu`, `swiglu`, `softplus`, `softmax`, `log_softmax`

**M2. Module / Linear / 초기화**
- `Module` 베이스 + `Sequential` / `ModuleList` / `ModuleDict`
- `state_dict()` / `load_state_dict()` (GPT-2 가중치 로드 호환)
- `Linear`, `Embedding`
- `nn.init`: `kaiming_uniform/normal`, `xavier_uniform/normal`, `normal_/uniform_/zeros_/ones_/constant_` + GPT-2 스타일

**M3. SGD → MNIST (첫 마일스톤)**
- `Optimizer` 베이스 + `SGD` (momentum, nesterov, weight_decay)
- `Dataset` + `DataLoader` (Node yield)
- MNIST loader (IDX 파싱)
- val acc 97%+, PyTorch parity ±1%

**M4. 디버깅/도구 (개발 편의)**
- **Named Dimensions** — 자세한 설계는 [NAMED_DIMS.md](./NAMED_DIMS.md)
  - `B, T, C = x.dims("B T C")` 로 shape 를 이름으로 bind, `assert_shape(B, T, C)` 로 검증
  - 4 단계: Dim 클래스 + `x.dims()` → `assert_shape` (Dim/int/None 혼용) → ShapeError 에 binding 위치 추적 → `DimExpr` + `__index__`/`__float__` (`C // H` 같은 산술이 reshape/슬라이싱에 직접 사용)
  - 일반 `Node.assert_shape(int, ...)` 는 이 기능의 부분집합으로 흡수
- `Net.eval_mode()` 컨텍스트, `check_gradients` (numerical), `Net.__repr__` tree, `Net.summary()`
- `Net.__matmul__` 으로 `@` 체이닝
- `Node.item()` (dtype 기반 Python 타입 추론)
- `checkpoint` (사람이 읽을 수 있는 npy 디렉토리 포맷)

**M5. 정규화 / 정칙화 / 고급 옵티마이저**
- `Adam` / `AdamW` / `Adagrad` / `RMSprop` / (선택) `Lion`
- LR scheduler: `StepLR`, `MultiStepLR`, `CosineAnnealingLR`, `LinearWarmup`, `WarmupCosine`
- `Dropout` / `Dropout2d`
- `BatchNorm1d/2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`

**M6. CNN / CIFAR (선택)**
- `Conv1d`, `Conv2d`, `ConvTranspose2d`
- `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- `Flatten`, `Identity`
- (선택) `ResidualBlock`, `Bottleneck` → ResNet-18 CIFAR-10

**M7. RNN / TinyShakespeare**
- `RNN`, `LSTM`, `GRU` cell + 시퀀스 wrapper
- 셰익스피어풍 텍스트 생성

**M8. Transformer / TinyShakespeare or TinyStories**
- `ScaledDotProductAttention`, `MultiHeadAttention`
- Causal mask, padding mask, KV cache
- Positional Encoding: Sinusoidal, Learned, RoPE
- `TransformerBlock` (decoder-only, GPT 스타일)
- Tokenization: char + BPE (`tiktoken` 또는 직접)
- Sampling: greedy / temperature / top-k / top-p / repetition penalty

**M9. GPT-2 124M inference (핵심 검증)**
- HF state_dict 다운로드 + 키 매핑
- logits 비교 (max abs diff < 1e-4)
- generation 결과 일치 (greedy / top-p)

**M10. Stretch**
- GPT-2 124M from-scratch 학습 (FineWeb / OpenWebText subset)
- ViT-Tiny CIFAR-10
- (선택) ALiBi, GQA, Flash Attention 흉내, SwiGLU FFN

각 Op / layer 끝나는 즉시 **numerical gradient check** + **PyTorch parity check** 로 검증.

---

## 파일 구조

```
axon/
├── axon/
│   ├── __init__.py
│   ├── backend/
│   │   ├── __init__.py     # set_backend, get_backend, xp, BackendProtocol
│   │   ├── _numpy.py
│   │   ├── _mlx.py
│   │   ├── _cupy.py
│   │   ├── _dtype.py       # backend native dtype ↔ axon DType 매핑
│   │   └── protocol.py     # Array, BackendProtocol
│   ├── dtype.py            # DType 클래스 계층
│   ├── scalar.py           # Scalar = int | float
│   ├── node.py             # Node — Array + 그래프 metadata
│   ├── parameter.py        # Parameter(Node) — 학습 leaf, .grad 버퍼
│   ├── backward.py         # backward(loss) — 위상 정렬 + grad dict
│   ├── operation/
│   │   ├── __init__.py
│   │   ├── op.py           # Op + UnaryOp + BinaryOp 베이스
│   │   ├── add.py, mul.py, matmul.py, div.py, neg.py
│   │   └── pow.py          # Pow + PowConstExp + PowConstBase 통합
│   ├── functional/         # 사용자 API (F.add, F.pow, ...) — Op dispatch
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── module.py       # Module, Sequential, ModuleList, ModuleDict
│   │   ├── linear.py
│   │   ├── conv.py
│   │   ├── pool.py
│   │   ├── norm.py         # BatchNorm, LayerNorm, RMSNorm
│   │   ├── dropout.py
│   │   ├── activation.py
│   │   ├── recurrent.py    # RNN, LSTM, GRU
│   │   ├── attention.py    # MHA, KV cache, mask
│   │   ├── pos_enc.py      # sinusoidal, RoPE, ALiBi
│   │   ├── transformer.py  # block, encoder, decoder
│   │   ├── loss.py
│   │   └── init.py
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── optimizer.py    # Optimizer 베이스 — Parameter 목록만 받음
│   │   ├── sgd.py
│   │   ├── adam.py         # Adam, AdamW
│   │   ├── rmsprop.py
│   │   └── scheduler.py
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── tokenizer.py    # char, BPE
│   └── generate.py         # sampling
├── examples/
│   ├── mnist.py
│   ├── cifar.py            # 선택
│   ├── resnet_cifar.py     # ResNet-18 검증
│   ├── shakespeare_lstm.py
│   ├── shakespeare_gpt.py
│   ├── tinystories_gpt.py
│   ├── gpt2_inference.py   # HF 가중치 로드 검증
│   └── vit_cifar.py        # 선택, stretch
├── tests/
│   ├── test_node.py
│   ├── test_autograd.py
│   ├── test_nn.py
│   ├── test_attention.py
│   ├── test_loss.py
│   ├── test_optim.py
│   ├── test_backends.py    # numpy ↔ mlx 결과 일치
│   └── test_pytorch_parity.py
├── pyproject.toml
└── README.md
```

---

## 의존 순서 (참고)

```
백엔드 추상화 (Array, BackendProtocol, xp) → DType / Scalar
→ Node (Array + 그래프 metadata) → Parameter (Node + grad)
→ Op (UnaryOp, BinaryOp) → 산술 Op → Autograd(backward)
→ 행렬/축소/broadcast Op → Activation (primitive 조합)
→ Module → Linear → 초기화
→ CrossEntropyLoss → SGD → MNIST
→ Adam/AdamW → Dropout → LayerNorm → RMSNorm
→ Embedding → MultiHeadAttention (mask, KV cache)
→ RoPE → TransformerBlock → Tokenizer → Sampling
→ TinyShakespeare GPT 학습 → GPT-2 가중치 로드 검증
→ TinyStories 학습 → (stretch) GPT-2 from-scratch
```

---

## 개발 환경

| 도구                                             | 역할             |
| ---------------------------------------------- | -------------- |
| [uv](https://github.com/astral-sh/uv)          | 패키지 매니저 + 가상환경 |
| [ruff](https://github.com/astral-sh/ruff)      | 린터 + 포매터       |
| [pyrefly](https://github.com/facebook/pyrefly) | 타입 체커 (Meta)   |
| [pytest](https://pytest.org)                   | 테스트            |

```toml
# pyproject.toml
[project]
name = "axon"
requires-python = ">=3.12"
dependencies = ["numpy"]

[project.optional-dependencies]
mlx = ["mlx"]
cuda = ["cupy-cuda12x"]
viz = ["matplotlib"]

[dependency-groups]
dev = ["pytest", "ruff", "pyrefly", "torch"]  # torch 는 검증 비교용

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 학습 자료

### 기초

- [3Blue1Brown — Neural Networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — 챕터 3, 4 (backprop 직관)
- [3Blue1Brown — Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) — chain rule
- [Karpathy — micrograd 영상](https://www.youtube.com/watch?v=VMj-3S1tku0) — 필수
- [cs231n — Backpropagation 노트](https://cs231n.github.io/optimization-2/) — 행렬 grad 공식

### Transformer

- [Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 자체 구현 영상. 핵심.
- [Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — 본격 학습 재현
- [nanoGPT](https://github.com/karpathy/nanoGPT) — reference 구현. 코드 비교용.
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — 원논문
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — 시각화 자료
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — 논문 + 코드 한 줄씩
- [3Blue1Brown — Attention 시리즈](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 챕터 5, 6, 7

### 백엔드

- [MLX 공식 문서](https://ml-explore.github.io/mlx/)
- [MLX-examples](https://github.com/ml-explore/mlx-examples) — 모델 구현 참고
