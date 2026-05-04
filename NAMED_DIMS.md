# axon — Named Dimensions

> `forward()` 안에서 shape를 이름으로 다루는 로컬 Dim 시스템.
> 전역 선언 없음. Python 스코프가 수명을 관리한다.

---

## 동기

```python
# 지금 — 숫자만 있어서 뭘 의미하는지 모름
def forward(self, x):
    x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
    scores = scores / 256 ** 0.5   # 256이 뭔지 알 수 없음

# 에러도 맥락 없음
# RuntimeError: Expected size 256 but got 512

# Named Dim — 의도가 코드에 있음
def forward(self, x):
    B, T, C = x.dims("B T C")
    q = q.reshape(B, T, H, C // H)
    scores = scores / C ** 0.5

# 에러도 맥락 있음
# ShapeError: dim C expected 256, got 512
#   assertion at  attention.py:47   q.assert_shape(B, T, C)
#   C was bound at attention.py:34  B, T, C = x.dims("B T C")
```

---

## API

### 생성 — `x.dims()`

```python
B, T, C = x.dims("B T C")
```

`x.shape`에서 즉시 bind. 이름 개수와 `x.ndim`이 다르면 즉시 에러.

```python
x = Tensor(shape=(32, 128, 256))

B, T, C = x.dims("B T C")
# B.value == 32
# T.value == 128
# C.value == 256

B, T = x.dims("B T")
# DimError: x is 3D but "B T" has 2 names
#   x.shape: (32, 128, 256)
```

---

### 검증 — `assert_shape()`

인자는 세 종류를 혼합할 수 있다.

```python
x.assert_shape(B, T, C)       # Dim  — bind or 검증
x.assert_shape(B, T, 256)     # int  — 고정값 검증
x.assert_shape(B, None, C)    # None — 해당 축 무시
```

`assert_shape()`는 `self`를 반환하므로 forward 중간에 인라인으로 체이닝할 수 있다.

```python
q = self.Wq(x).assert_shape(B, T, C)
k = self.Wk(x).assert_shape(B, T, C)
v = self.Wv(x).assert_shape(B, T, C)
```

**Dim이 unbound일 때** — 현재 축의 크기로 bind하고 위치를 기록한다.

**Dim이 already bound일 때** — 기록된 값과 비교해 다르면 `ShapeError`.

```python
x.assert_shape(B, T, C)   # C → 256으로 bind

out = linear(x)
out.assert_shape(B, T, C)  # out.shape[-1] == 512 이면:
# ShapeError: dim C expected 256, got 512
#   assertion at  attention.py:47   out.assert_shape(B, T, C)
#   C was bound at attention.py:34  B, T, C = x.dims("B T C")
#   actual:   (32, 128, 512)
#   expected: (B=32, T=128, C=256)
```

---

### DimExpr — Dim 산술

Dim끼리, 또는 Dim과 정수 사이의 연산은 `DimExpr`를 반환한다.
구성 요소가 bind된 시점에 자동으로 평가된다.

```python
B, T, C = x.dims("B T C")
H = 8            # 일반 int
D = C // H       # DimExpr — 아직 평가되지 않음
                 # (C가 이미 bind됐으면 즉시 32로 평가)
```

지원 연산:

```python
C // H    # → DimExpr(floordiv, C, 8)
C * 4     # → DimExpr(mul, C, 4)
C + 1     # → DimExpr(add, C, 1)
C - 1     # → DimExpr(sub, C, 1)
C ** 0.5  # → DimExpr(pow, C, 0.5)
```

#### `__index__` / `__float__` — 정수/float가 필요한 자리에서 자동 평가

`__index__`를 구현하면 Python이 정수를 요구하는 모든 자리에서 자동으로 `int(dim)`을 호출한다.

```python
# reshape — shape tuple 안의 __index__ 자동 호출
q = x.reshape(B, T, H, D)          # (32, 128, 8, 32)

# 슬라이싱
x[:, :T, :]                         # T가 정수로 평가됨

# xp 함수 — shape tuple
xp.zeros((B, T, C))                 # (32, 128, 256)

# 나눗셈 스케일 팩터 — __float__ 호출
scores = scores / C ** 0.5          # / 16.0
```

unbound Dim에서 `__index__` / `__float__`가 호출되면 즉시 에러.

```python
D = C // H   # C가 아직 unbound

x.reshape(B, T, H, D)
# DimError: Dim 'C' is not bound yet
#   D = C // H defined at attention.py:12
#   reshape called at attention.py:38
#   hint: call x.dims() or x.assert_shape() before using C
```

---

## 에러 메시지

### 바인딩 위치 추적

`x.dims()` 와 `assert_shape()` 호출 시점에 파일명과 라인을 기록한다.

에러가 나면 세 줄로 맥락을 제공한다.

```
ShapeError: dim C expected 256, got 512
  assertion at  attention.py:47   q.assert_shape(B, T, C)
  C was bound at attention.py:34  B, T, C = x.dims("B T C")
  actual:   (32, 128, 512)
  expected: (B=32, T=128, C=256)
```

### ndim 불일치

```
ShapeError: expected 4D tensor (B, T, H, D) but got 3D
  assertion at  attention.py:52   q.assert_shape(B, T, H, D)
  actual shape: (32, 128, 256)
```

### DimExpr 정수 불가

```
ShapeError: C // H must divide evenly, got 256 // 9 = 28.4...
  D = C // H used at  attention.py:42   q.reshape(B, T, H, D)
  C=256 bound at      attention.py:34   B, T, C = x.dims("B T C")
  H=9
```

---

## 완성 예시 — MultiHeadAttention.forward

```python
class MultiHeadAttention(Module):
    def __init__(self, dim: int, n_heads: int):
        self.H = n_heads
        self.Wq = Linear(dim, dim)
        self.Wk = Linear(dim, dim)
        self.Wv = Linear(dim, dim)
        self.Wo = Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.dims("B T C")
        H, D = self.H, C // H

        q = self.Wq(x).assert_shape(B, T, C).reshape(B, T, H, D)
        k = self.Wk(x).assert_shape(B, T, C).reshape(B, T, H, D)
        v = self.Wv(x).assert_shape(B, T, C).reshape(B, T, H, D)
        # q, k, v: (B, T, H, D)

        q = q.transpose(1, 2).assert_shape(B, H, T, D)
        k = k.transpose(1, 2).assert_shape(B, H, T, D)
        v = v.transpose(1, 2).assert_shape(B, H, T, D)

        scores = (q @ k.transpose(-2, -1)) / C ** 0.5
        scores.assert_shape(B, H, T, T)

        out = (F.softmax(scores, axis=-1) @ v).assert_shape(B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)

        return self.Wo(out).assert_shape(B, T, C)
```

---

## 구현 순서

```
1단계  Dim 클래스 + x.dims()
       x.shape에서 즉시 bind, 이름 개수 불일치 에러

2단계  assert_shape(Dim, int, None 혼용)
       bind 로직, 바인딩 위치 기록 (inspect)

3단계  에러 메시지 — 바인딩 위치 포함한 ShapeError

4단계  DimExpr + __index__ / __float__
       C // H 산술, reshape / xp.zeros에 Dim 직접 사용
```

---

## 범위 밖 (일단 제외)

| 기능 | 이유 |
|---|---|
| `freeze()` / `frozen_dims` | 로컬이면 forward 종료 시 자동 소멸 — 불필요 |
| Constraint (`must_be_divisible_by` 등) | 타입 어노테이션 없이는 가치 절반 — 나중에 |
| `dim_scope()` 자동 스코프 | 로컬이면 Python 스코프가 수명 관리 |
| 전역 공유 | 로컬로 고정 |
| `Tensor[B, T, C]` 제네릭 | pyrefly 연동 필요 — 나중에 |
