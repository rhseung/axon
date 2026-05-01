# axon

> NumPy 기반 자동 미분 엔진 + 신경망 프레임워크. MNIST 분류까지.

---

## 목표

- Autograd 직접 구현 (computation graph + chain rule)
- PyTorch / Flax NNX에서 좋은 부분만 가져온 API
- GPU 지원 가능한 구조 (NumPy ↔ CuPy 교체 가능)
- MNIST 분류 동작 확인

---

## API 설계 방향

```python
# 모델 정의 — PyTorch 스타일
class MLP(nn.Module):
    def __init__(self):
        self.l1 = nn.Linear(784, 256)
        self.l2 = nn.Linear(256, 10)

    def forward(self, x):
        x = axon.relu(self.l1(x))
        return self.l2(x)

# grad — Flax/JAX 스타일 (명시적)
loss, grads = axon.value_and_grad(loss_fn)(model, x, y)
optimizer.update(grads)
```

---

## 구현 계획

### Phase 0 — 공부

**수학 / 직관**

- [3Blue1Brown — Neural Networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — 챕터 3, 4가 핵심. "What is backpropagation really doing?"이랑 "Backpropagation calculus" 두 편.
- [3Blue1Brown — Essence of Calculus](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) — chain rule 챕터만 봐도 됨.

**구현 직접 참고**

- [Andrej Karpathy — micrograd 구현 영상](https://www.youtube.com/watch?v=VMj-3S1tku0) — 2.5시간짜리. 지금 하려는 거랑 거의 완전히 동일한 걸 처음부터 구현함. 필수.
- [Karpathy — Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — micrograd 영상이 첫 번째. 이후는 언어모델 쪽이라 지금 단계에선 선택.

**읽을 것**

- [cs231n — Backpropagation 노트](https://cs231n.github.io/optimization-2/) — 행렬 연산 역전파 공식이 잘 정리되어 있음. `matmul` grad 구현할 때 참고.

---

### Phase 1 — Tensor & Autograd (핵심)

- [ ] `Tensor` 클래스
  - `data` (ndarray), `grad`, `_backward`, `_prev`
  - `device` 필드 + NumPy/CuPy backend 선택
- [ ] 스칼라 연산부터 시작
  - `__add__`, `__mul__`, `__pow__`, `__neg__`
- [ ] `backward()` 구현
  - 위상 정렬 (topological sort)
  - loss → input 방향으로 `_backward` 순서대로 호출
  - 같은 텐서가 여러 번 쓰일 때 grad 합산
- [ ] 배열 연산으로 확장
  - `__matmul__` — grad shape 맞추기 (`A.T @ g`, `g @ B.T`)
  - `sum`, `mean` — axis별 grad 복원
  - broadcast된 연산의 grad — `np.sum`으로 축 맞춤
- [ ] Activation
  - `relu`, `sigmoid`, `tanh`, `exp`, `log`
- [ ] `value_and_grad(f)` 함수 변환 인터페이스

### Phase 2 — nn 추상화

- [ ] `Module` 베이스 클래스
  - `parameters()` — 하위 Module의 Tensor를 재귀적으로 수집
  - `__call__` → `forward()` 호출
  - `zero_grad()` — 모든 파라미터의 grad 초기화
- [ ] `Linear(in, out)` — weight, bias 초기화 포함
- [ ] `Sequential` (선택)

### Phase 3 — Loss & Optimizer

- [ ] Loss
  - `MSELoss`
  - `CrossEntropyLoss` — numerically stable softmax + NLL
- [ ] `SGD` — `param.data -= lr * param.grad`
- [ ] `Adam` — moment 상태 (`m`, `v`) 관리, bias correction

### Phase 4 — MNIST

- [ ] 데이터 로딩 (torchvision 또는 직접 파싱)
- [ ] 학습 루프 작성
- [ ] 정확도 측정
- [ ] 목표: val accuracy 97%+

---

## 개발 환경

| 도구 | 역할 |
|---|---|
| [uv](https://github.com/astral-sh/uv) | 패키지 매니저 + 가상환경. `pip` + `venv` 대체 |
| [ruff](https://github.com/astral-sh/ruff) | 린터 + 포매터. `flake8` + `black` 대체 |
| [pyrefly](https://github.com/facebook/pyrefly) | 타입 체커. Meta가 만든 빠른 type checker |
| [pytest](https://pytest.org) | 테스트 프레임워크. numerical gradient check 검증에 사용 |

```toml
# pyproject.toml
[project]
name = "axon"
requires-python = ">=3.12"
dependencies = ["numpy"]

[dependency-groups]
dev = ["pytest", "ruff", "pyrefly"]

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
│   ├── __init__.py         # relu, value_and_grad 등 top-level export
│   ├── tensor.py           # Tensor 클래스 + autograd engine
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── module.py       # Module 베이스
│   │   ├── layers.py       # Linear, Sequential
│   │   └── loss.py         # MSELoss, CrossEntropyLoss
│   └── optim/
│       ├── __init__.py
│       ├── sgd.py
│       └── adam.py
├── examples/
│   └── mnist.py
├── tests/
│   ├── test_tensor.py      # grad 수치 검증 (numerical gradient check)
│   └── test_nn.py
├── pyproject.toml
└── README.md
```

---

## 검증 방법

자동 미분 구현 후 **numerical gradient check**로 반드시 검증.

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

구현한 `grad`값이랑 이 수치 미분값이 거의 일치하면 (오차 < 1e-5) 정확한 것.

---

## 구현 순서 요약

```
Tensor (스칼라) → backward() → 배열 연산 → value_and_grad
→ Module / Linear → CrossEntropyLoss → SGD → MNIST
→ Adam 추가
```
