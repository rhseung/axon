# axon

자동미분이 어떻게 동작하는지 직접 부딪치며 만들어보는 토이 프로젝트예요.
공부 목적이라 PyPI 배포 안 하고, production 사용 안 해요. "이렇게
만들어봤어요" 의 기록이고, 코드 읽고 따라 만들어보고 싶은 사람을 위한
reference 예요. 설계 결정의 이유는 [`CLAUDE.md`](./CLAUDE.md) 에 길게
적어뒀어요.

## 왜 만드나

신경망 라이브러리를 그냥 갖다 쓰기만 하다 보니 — 이게 안에서 뭘 하는지
알고 싶더라고요. backward 가 어떻게 흐르는지, optimizer state 가 어디
사는지, MLX / NumPy 가 같은 코드로 어떻게 굴러가는지.

그래서 직접 만들어보는 중이에요. axon 은 그 학습 일지 같은 거예요.

NumPy / MLX / CuPy 셋을 같은 인터페이스 뒤로 숨겨서, 같은 코드가 세
백엔드 다 돌아가요. M 시리즈 (Apple Silicon) 에서 MLX 로 GPU 활용도
한 줄 — `axon.set_backend("mlx")`.

## 어떻게 써보나

```bash
git clone https://github.com/<user>/axon.git
cd axon
uv sync                           # 가상환경 + 의존성
uv sync --extra mlx               # Apple Silicon GPU
uv sync --extra cuda              # NVIDIA GPU
```

Python 3.12 이상이 필요해요.

## 5분 예제

MNIST 학습 한 step.

```python
import axon
from axon import Var, Constant
import axon.backend as xp
import axon.net as net
import axon.functional as F
from axon.optim import AdamW

axon.set_backend("mlx")           # 또는 "numpy", "cupy"

class MLP(net.Net):
    optimizer = AdamW(lr=3e-4)    # 모든 학습 weight 의 default

    def __init__(self):
        self.fc1 = net.Linear(784, 128)
        self.fc2 = net.Linear(128, 10)

    def forward(self, x: Constant) -> Var:
        h = F.relu(self.fc1(x))
        return self.fc2(h)

model = MLP()
loss_fn = net.CrossEntropyLoss()

for x_batch, y_batch in train_loader:
    logits = model.forward(Constant(x_batch))
    loss = loss_fn(logits, Constant(y_batch))

    loss.backward()
    loss.optimize()               # update + grad zero, 한 줄
```

`set_backend("numpy")` 로 바꾸면 같은 코드가 NumPy 로 돌아가요.

PyTorch 의 `optimizer.zero_grad() / loss.backward() / optimizer.step()`
세 줄 패턴이 axon 에선 두 줄로 줄어들어요. optimizer 는 별도 객체로
떠다니지 않고 학습 weight 의 일부예요. 자세한 건 아래 "axon 의 결정들"
에서.

## 핵심 API

axon 은 다섯 레이어로 자료가 흘러가요:

```
Array              순수 데이터. np.ndarray / mx.array / cp.ndarray.
Constant / Var     그래프 노드. 비추적 / 추적.
Op                 연산 정의 (forward + backward). 비공개.
functional         순수 연산 함수. F.add, F.matmul, F.relu, ...
net                Net base + 구체 module (Linear, ReLU, LayerNorm, CrossEntropyLoss, ...)
```

### Array — 순수 데이터

DataLoader 가 던져주는 배치, weight 의 초기값, 체크포인트로 저장하는
값은 전부 Array 예요. axon 만의 별도 클래스를 만들지 않고, 백엔드의
ndarray 를 그대로 써요.

```python
import axon.backend as xp

x = xp.array([[1.0, 2.0], [3.0, 4.0]])    # 현재 백엔드의 array
w = xp.random.normal((128, 64), std=0.01)
z = xp.zeros((10,))
```

### Constant / Var — 그래프 노드

`Constant` 는 비추적 (입력 데이터, 상수, KV cache 같은 거). `Var` 는
추적 (학습 weight, 중간 결과). 수학 표기 $L = f(x; W, b)$ 에서 $x$ 는
상수, $W, b$ 는 변수 — 미분 대상의 구분이 그대로 정적 타입에 들어가요.

```python
x = Constant(x_batch)             # 비추적, grad 안 흐름
W = Var(xp.random.normal((784, 128)))    # 추적, grad 흐름
W = Var(weight_init, optimizer=AdamW(lr=1e-3))   # 자기 update 정책 부착
```

`Constant` 엔 `grad` 필드도, `backward` / `step` 메서드도 없어요.
잘못 쓰면 정적 타입에서 즉각 잡혀요:

```python
x = Constant([1, 2, 3])
x.grad             # 정적 타입 에러
x.backward()       # 정적 타입 에러
```

`Var` 끼리 또는 `Var` 와 `Constant` 의 chain 결과는 자동으로 `Var`.
`Constant` 끼리만 chain 하면 결과도 `Constant`.

### functional — 순수 연산

```python
import axon.functional as F

y = F.add(a, b)                # 또는 a + b (v1)
y = F.matmul(x, W)             # 또는 x @ W (v1)
y = F.relu(x)
y = F.log_softmax(logits)
```

functional 은 state 가 없는 순수 연산이에요. 한 번 호출하면 끝, 인스턴스
유지 안 해요. forward 안에서 inline 으로 부르는 게 자연스러워요 —
`F.relu(self.fc1(x))` 처럼.

연산자 오버로딩 (v1) 도 지원해서 `x @ W + b` 처럼 자연스럽게 쓸 수
있어요. Python scalar (int, float) 는 자동으로 `Constant` 로 lift 돼서
`x * 2.0` 같은 표현도 바로 동작해요.

### loss.backward() / loss.optimize()

학습 루프의 두 줄.

```python
loss = F.cross_entropy(logits, labels)

loss.backward()    # 그래프 따라 grad 누적
loss.optimize()        # 학습 weight update + grad zero
```

`backward` 는 그래프 책임 — 모든 추적 노드의 grad 를 계산. `step` 은
weight 책임 — 그래프 도달 가능한 학습 weight 들이 자기 optimizer 로
update 하고 grad 를 0 으로.

`Var` 는 모두 `.grad` 필드를 가지니까 디버깅할 때 중간 노드의 grad 도
바로 들여다볼 수 있어요.

**Gradient accumulation** — `step` 만 빼고 `backward` 여러 번:

```python
for micro in micro_batches:
    compute_loss(micro).backward()    # grad 누적만
final_loss.optimize()                      # 마지막에 한 번만 update
```

### net — Net base + 구체 module

`axon.net` 패키지에 base `Net` 클래스와 자주 쓰이는 module 들이 다
들어 있어요. PyTorch 의 `nn` 패키지와 비슷한 구조.

```python
import axon.net as net

# 자주 쓰는 것들
net.Linear(in_dim, out_dim)
net.LayerNorm(dim)
net.RMSNorm(dim)
net.Embedding(vocab, dim)
net.Dropout(p=0.1)

# 활성화 — Sequential 에 끼우거나 인스턴스로 보관
net.ReLU()
net.GELU()
net.SiLU()

# Loss
net.CrossEntropyLoss(ignore_index=-100)
net.MSELoss()

# 구조
net.Sequential(...)

# 복합
net.MultiHeadAttention(dim, n_heads)
net.TransformerBlock(dim, n_heads)
```

직접 `Net` 자식 만들 때:

```python
class TransformerBlock(net.Net):
    optimizer = AdamW(lr=3e-4, weight_decay=0.01)

    def __init__(self, dim, n_heads):
        self.norm = net.LayerNorm(dim)
        self.attn = net.MultiHeadAttention(dim, n_heads)
        self.ffn  = FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(self.norm(x))
        return x
```

`__dict__` 재귀 순회로 `parameters()` 가 학습 weight 를 자동 수집해요.
`net.Net.optimizer` class attribute 가 default 라서 `__init__` 에서 그냥
선언만 해두면 자동으로 AdamW 가 부착돼요. 필드에 그냥 선언만 해두면
등록은 알아서 돼요.

#### functional vs net 의 경계

| | functional | net |
|---|---|---|
| state | 없음 | 있을 수도 (weight, 옵션) |
| 호출 | 함수 | 인스턴스 메서드 |
| 위치 | forward 안 inline | `__init__` 에서 인스턴스화 |
| 활성화 | `F.relu(x)` | `net.ReLU()` (Sequential 용) |
| Loss | `F.cross_entropy(logits, y)` | `net.CrossEntropyLoss()` (옵션 보관) |

활성화 / loss 는 양쪽 다 있어요. inline 호출이 자연스러우면 `F.relu(x)`,
Sequential 에 끼워넣거나 옵션 (`ignore_index`, `reduction` 등) 을 보관하고
싶으면 `net.ReLU()` / `net.CrossEntropyLoss(...)`.

Linear / LayerNorm 같은 weight 가지는 layer 는 항상 `net` 만 있어요 —
state 가 인스턴스에 사니까 함수형으로 표현 안 됨.

#### Layer-wise LR (LLRD)

```python
class GPT(net.Net):
    def __init__(self, n_layers=12):
        # i 번째 layer 가 LR scale 0.9^(n-i)
        self.blocks = [
            TransformerBlock(dim=768, lr_scale=0.9 ** (n_layers - i))
            for i in range(n_layers)
        ]
```

각 layer 가 자기 optimizer 를 가지고, 그래프의 모든 weight 가 자기
정책대로 update 돼요. PyTorch 의 `param_groups` 같은 별도 매커니즘 없이
layer 클래스 안에서 자연스럽게 표현.

#### Net 글로벌 동작

Gradient clipping 같은 모든 weight 에 한 번에 적용하는 동작은 Net
메서드:

```python
loss.backward()
model.clip_grad_norm(max_norm=1.0)    # 전체 grad norm 정규화
loss.optimize()
```

`model.grad_norm()`, `model.zero_grad()` 도 같은 패턴.

### 백엔드 전환

```python
axon.set_backend("mlx")        # Apple Silicon GPU
axon.set_backend("numpy")      # CPU, 기준 구현
axon.set_backend("cupy")       # NVIDIA GPU
```

같은 코드가 세 백엔드 모두에서 동작해요. NumPy ↔ MLX 결과 일치는 parity
테스트로 보장하고요.

## 어디까지 만들 거예요

대략 이런 단계로 갈 생각이에요. 실제로 어디까지 갈진 모르겠고, 학습이
계속되는 한 천천히 채워질 거예요.

| 단계 | 내용 | 상태 |
|---|---|---|
| v0 | 자동미분 엔진 (Add/Mul/MatMul/Pow 등) | ⏳ 마무리 중 |
| v1 | Net + Optimizer + MNIST 97% | |
| v2 | LayerNorm / Embedding / Dropout + LSTM TinyShakespeare | |
| v3 | Attention + Transformer + TinyShakespeare GPT | |
| v4 | GPT-2 124M 가중치 로드 + HF 와 logits 일치 | |
| v5 | (Stretch) GPT-2 from scratch | |

상세는 [`CLAUDE.md`](./CLAUDE.md) §4 단계별 로드맵 참고.

## axon 의 결정들

axon 은 다른 라이브러리를 흉내 내려고 만든 게 아니라, 자동미분이 어떻게
동작하는지 직접 부딪히면서 답을 찾아가는 프로젝트예요. 몇 가지 결정이
다른 라이브러리와 달라요.

**Var / Constant 두 클래스로 분리**. 한 `Tensor` 클래스에 `requires_grad`
플래그를 두는 PyTorch 패턴 대신, 미분 대상 (`Var`) 과 비대상
(`Constant`) 을 정적 타입으로 분리. 잘못된 사용 (`Constant.backward()`
같은 거) 이 컴파일 시점에 잡혀요. 수학에서 변수와 상수의 구분이 그대로
타입에 옮겨진 거예요.

**Optimizer 가 weight 의 일부**. PyTorch 의 `Optimizer` 객체가 별도로
떠다니는 패턴이 axon 엔 없어요. 학습 weight 인 `Var` 가 자기 update
정책을 직접 들고 있고, Adam 의 m, v 같은 state 도 그 weight 의 일부로
같이 살아요. 학습 루프는 `loss.backward(); loss.optimize()` 두 줄이고,
layer-wise LR 이나 weight-decay 분기 같은 흔한 워크로드도 layer 클래스
안에서 자연스럽게 표현돼요.

**Parameter 클래스 없음**. 학습 weight 는 그냥 leaf 인 `Var` 일 뿐
(`is_parameter` property 가 `_op is None` 한 줄). 마커 클래스를 따로
두지 않고 그래프 metadata 만으로 정의해요.

**Op 은 비공개**. 사용자는 `axon.functional` 한 길로만 연산해요. 같은
일을 두 길로 할 수 있으면 어느 게 표준인지 헷갈리거든요.

**연산자 오버로딩은 `_coerce_pair` 로 한 곳에서 lifting**. backend Array
와 그래프 노드를 섞는 건 명시적으로 거부해요 — 백엔드마다 dispatch
매커니즘이 달라서 (`__array_priority__` 는 NumPy 만 봐요), 자동으로
받아들이면 조용한 버그가 잘 생기거든요.

**`Pow` 만 dispatch 패턴**. `x ** 2` 같은 흔한 케이스에서 `Pow.backward`
의 `xp.log(a)` 항이 a < 0 일 때 NaN 을 뿌려요. 그래서 지수 / 밑이 상수면
별도 Op (`PowConstExp`, `PowConstBase`) 으로 분기해서 위험한 산수를
회피해요.

자세한 설계 메모는 [`CLAUDE.md`](./CLAUDE.md) 에 있어요.

## 개발

```bash
# 가상환경
uv sync

# 테스트
pytest

# 린트 / 포맷
ruff check
ruff format

# 타입 체크
pyrefly check
```

## 라이선스

MIT.
