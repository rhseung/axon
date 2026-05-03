# axon

> 자동 미분 엔진 + 신경망 프레임워크. **GPT-2 학습/추론까지.**

---

## 목표

- Autograd 직접 구현 (computation graph + chain rule)
- PyTorch / Flax NNX에서 좋은 부분만 가져온 API
- 백엔드 추상화: **NumPy** (CPU) / **MLX** (Apple Silicon Metal) / **CuPy** (NVIDIA, 옵션)
- 검증 단계별 마일스톤
  - MLP — MNIST val 97%+
  - CNN — CIFAR-10 (선택)
  - RNN/LSTM — TinyShakespeare char-level
  - **Transformer — TinyStories or TinyShakespeare BPE 학습**
  - **GPT-2 124M 가중치 로드 → inference 결과가 HuggingFace와 동일**

---

## 설계 철학

### Tensor는 순수한 값이다

`Tensor`는 값(data)과 연산 그래프 정보(`_op`, `_inputs`)만 들고 있다. grad는 없다.
grad는 `backward(loss)`의 내부 `grads` dict 안에서만 흐르고, `Parameter`일 때만 바깥으로 누적된다.

```
Tensor   — data + _op + _inputs   (순수한 값, 그래프 노드)
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

grad를 `inputs[i].grad`에 직접 쓰지 않는다. `backward(loss)`가 반환값을 받아 `Parameter`인 경우에만 누적한다.

### 학습 루프

```python
model = Sequential([
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
])

optimizer = SGD(model.parameters(), lr=0.01)
loss_fn = CrossEntropyLoss()

for x, t in dataloader:
    logits = model(Tensor(x))   # forward — 순수한 값의 흐름
    loss = loss_fn(logits, t)

    optimizer.zero_grad()       # Parameter.grad = 0
    backward(loss)              # grad는 Parameter에만 누적
    optimizer.step()            # Parameter.data -= lr * Parameter.grad
```

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
optimizer.zero_grad()
backward(loss)
optimizer.step()

# 백엔드 선택
axon.set_backend("mlx")  # or "numpy", "cupy"
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

- `Tensor.to(device)` — `"cpu"`, `"gpu"`, `"mps"`
- CPU↔GPU 자동 변환 막기 (성능 함정)

---

## Tensor

> **Tensor는 순수한 값이다.** grad, requires_grad 없음.

- `data` (backend array), `_op`, `_inputs` 필드
- `device`, `dtype` 필드
- `shape`, `ndim` 프로퍼티
- `detach()` — graph에서 분리 (Identity op으로 끊기)
- `to(device)`, `to(dtype)`
- `numpy()` — 변환
- `__repr_`_ — shape + 일부 값

---

## Op

- `Op` 추상 클래스 — `forward(*inputs) -> Tensor`, `backward(grad, inputs) -> tuple[NDArray, ...]`
- `backward`는 각 input에 전달할 grad를 **반환**한다 — side effect 없음
- `_make_output(inputs, output)` — `_op`, `_inputs` 연결 헬퍼
- `Identity` — 리프 노드용 no-op

---

## 연산 (forward + backward)

### 산술

- `__add__`, `__sub__`, `__mul__`, `__truediv__`
- `__pow__`, `__neg__`
- `__radd__`, `__rmul__` 등 reverse
- `__matmul__`

### 축소 / 형상

- `sum`, `mean`, `max`, `min`, `var`, `std`
- `reshape`, `view`, `transpose`, `permute`
- `squeeze`, `unsqueeze`, `expand`
- `concat`, `stack`, `split`, `chunk`

### 인덱싱

- `__getitem__` — slice / int / fancy indexing / bool mask
- `gather`, `scatter`
- `where` (마스킹용)

### 수학 함수

- `exp`, `log`, `sqrt`, `rsqrt`
- `abs`, `sign`, `clip`
- `sin`, `cos` (RoPE에 필요)

---

## Autograd

- 위상 정렬로 노드 순서 결정
- `backward(loss)` — `grads: dict[id, NDArray]` 내부에서 grad 흐름
  - `Op.backward`가 반환한 grad를 각 input에 합산
  - `isinstance(inp, Parameter)`일 때만 `inp.grad +=` 누적
- grad 합산 — 같은 텐서가 여러 경로로 쓰일 때 `grads[id]` 에서 자동 누적
- Broadcasting 후 broadcast된 축 복원
- `keepdims` 처리
- `no_grad()` 컨텍스트 매니저 — `_inputs` 기록 생략
- **Gradient checkpointing** — 메모리 위해 일부 노드 재계산
- **Gradient clipping** — `clip_grad_norm_`

---

## Activation

- `relu`
- `leaky_relu` (negative_slope 고정값)
- `prelu` — α가 학습 파라미터, `nn.PReLU` Module + 함수 둘 다
- `elu`
- `gelu` (exact + tanh approx)
- `silu` / `swish`
- `mish`
- `glu` — Gated Linear Unit
- `swiglu` — LLaMA FFN에서 사용
- `sigmoid`, `tanh`
- `softmax`, `log_softmax` (수치 안정)
- `softplus`

---

## Layers (`nn`)

### 베이스

- `Module` — `parameters()`, `zero_grad()`, `train()`, `eval()`
  - `parameters()` — `__dict_`_를 재귀 순회해 `Parameter` 수집
  - `zero_grad()` — `p.grad = zeros_like(p.data)` for p in parameters()
- `Parameter` — `Tensor`의 서브클래스. `grad: NDArray` 필드 추가. 옵티마이저와의 유일한 접점.
- `Sequential`, `ModuleList`, `ModuleDict`
- `state_dict()`, `load_state_dict()` — 가중치 저장/로드 (HF 호환 위해 필수)

### Linear / Embedding

- `Linear(in, out, bias=True)`
- `Embedding(num_emb, dim)`

### Convolution

- `Conv1d`, `Conv2d`
- `ConvTranspose2d`

### Pooling

- `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`

### Normalization

- `BatchNorm1d`, `BatchNorm2d` — running stats, train/eval
- `LayerNorm`
- `RMSNorm` — LLaMA 계열
- `GroupNorm`
- `InstanceNorm1d`, `InstanceNorm2d` (선택)

### Regularization

- `Dropout` — train/eval 분기
- `Dropout2d` (CNN용)

### Recurrent

- `RNN` cell + 시퀀스 wrapper
- `LSTM`
- `GRU`

### Residual / Skip

- **Residual connection** — `x + sublayer(x)` 패턴, Transformer/ResNet 공통
- `ResidualBlock` (BasicBlock) — Conv-BN-ReLU-Conv-BN + skip, ResNet-18/34용
- `Bottleneck` — 1×1 → 3×3 → 1×1 + skip, ResNet-50/101/152용
- `SqueezeExcitation` (선택) — SE-Net 채널 어텐션

### Attention

- `ScaledDotProductAttention` — 함수, `softmax(QK^T/√d)V` 수식만
- `SingleHeadAttention` — Module, q/k/v/o projection 포함, head 1개 (학습/디버깅 단계용)
- `MultiHeadAttention` — q/k/v/o projection + head split/merge
- **Causal mask** — `tril`, `-inf` 채우기
- **Padding mask**
- **KV cache** — 추론 시 이전 k/v 재사용
- **Grouped Query Attention** (선택, LLaMA-2 70B)
- **Flash Attention 흉내** — 진짜 fused는 어렵고, 메모리 효율적인 청크 버전 정도 (선택)

### Positional Encoding

- **Sinusoidal PE** — 원논문 스타일
- **Learned PE** — GPT-2 스타일
- **RoPE (Rotary Position Embedding)** — LLaMA 스타일
- **ALiBi** (선택)

### Transformer 블록

- `TransformerEncoderLayer` — Self-Attn + FFN + 2× Norm (Pre-norm)
- `TransformerDecoderLayer` — + Cross-Attn (선택)
- `TransformerBlock` (GPT 스타일, decoder-only)
- FFN 변형: 일반 (Linear-GELU-Linear) / SwiGLU (LLaMA)

### 기타

- `Flatten`
- `Identity`

---

## Loss

- `MSELoss`, `L1Loss`, `HuberLoss`
- `CrossEntropyLoss` — stable log_softmax + NLL, ignore_index 옵션
- `NLLLoss`
- `BCELoss`, `BCEWithLogitsLoss`
- **Label smoothing** — CE 옵션
- **Perplexity** — eval 지표 (loss → exp)

---

## Optimizer

### 베이스

- `Optimizer(params: list[Parameter])` — `step()`, `zero_grad()`, param group, state dict
- `step()` — `Parameter.data`를 `Parameter.grad`로 갱신
- `zero_grad()` — `Parameter.grad = zeros_like`

### 구현

- `SGD` — momentum, nesterov, weight_decay
- `Adagrad`
- `RMSprop`
- `Adam` — m, v + bias correction
- `AdamW` — decoupled weight decay (Transformer 표준)
- `Lion` (선택)

### LR Scheduler

- `StepLR`, `MultiStepLR`
- `CosineAnnealingLR` — Transformer 학습에 자주 쓰임
- `LinearWarmup` — 처음 N step 워밍업
- `WarmupCosine` — 위 둘 합성

---

## 초기화 (`nn.init`)

- `xavier_uniform`, `xavier_normal`
- `kaiming_uniform`, `kaiming_normal`
- `normal`_, `uniform_`
- `zeros_`, `ones_`, `constant_`
- **GPT-2 스타일** — `std=0.02` normal, residual projection은 `1/sqrt(2*n_layer)` 스케일

---

## Tokenization

- **Character-level** — 가장 간단, TinyShakespeare 학습용
- **Byte-level BPE (GPT-2 호환)** — `tiktoken` 또는 직접 BPE 학습
- vocab 저장/로드
- HuggingFace tokenizer 호환 (옵션) — GPT-2 가중치 로드 검증용

---

## Sampling / Generation

- **Greedy decoding**
- **Temperature**
- **Top-k**
- **Top-p (nucleus)**
- **Repetition penalty** (선택)
- **KV cache 활용한 incremental decoding**
- 스트리밍 출력

---

## 데이터 / 학습 인프라

### 데이터

- `Dataset` 베이스
- `DataLoader` — 배치, 셔플, num_workers (멀티프로세싱은 선택)
- **MNIST** loader (IDX 파싱)
- **CIFAR-10** loader (선택)
- **TinyShakespeare** loader (텍스트 → 토큰 stream)
- **TinyStories** loader

### 학습 유틸

- 학습 루프 표준화
- **Mixed precision** — fp32 master + fp16/bf16 forward (MLX는 bf16 native)
- **Gradient accumulation** — 작은 GPU에서 큰 batch 흉내
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
- CIFAR-10에서 90%+ 또는 Imagenette 등 작은 이미지넷 subset에서 검증
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
- PyTorch와 forward/backward 결과 비교 (같은 입력 → 같은 출력 / grad)
- 백엔드별 동일 결과 검증 (numpy ↔ mlx)
- toy 문제로 옵티마이저 수렴 확인
- state_dict 저장 후 로드해서 결과 동일

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
dev = ["pytest", "ruff", "pyrefly", "torch"]  # torch는 검증 비교용

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## 파일 구조

```
axon/
├── axon/
│   ├── __init__.py
│   ├── backend/
│   │   ├── __init__.py     # set_backend, get_backend
│   │   ├── numpy_backend.py
│   │   ├── mlx_backend.py
│   │   └── cupy_backend.py
│   ├── tensor.py           # Tensor, Parameter, backward()
│   ├── op.py               # Op 추상 클래스 + Identity
│   ├── ops.py              # 산술/축소/형상 Op 구현
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
│   ├── test_tensor.py
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
백엔드 추상화 → Tensor / Parameter → Op 추상 클래스
→ 산술 Op → Autograd(backward) → 행렬/축소/broadcast Op
→ Activation → Module / Parameter → Linear → 초기화
→ CrossEntropyLoss → SGD → MNIST
→ Adam/AdamW → Dropout → LayerNorm → RMSNorm
→ Embedding → MultiHeadAttention (mask, KV cache)
→ RoPE → TransformerBlock → Tokenizer → Sampling
→ TinyShakespeare GPT 학습 → GPT-2 가중치 로드 검증
→ TinyStories 학습 → (stretch) GPT-2 from-scratch
```

각 레이어/연산 끝나는 즉시 numerical grad check + PyTorch parity check.