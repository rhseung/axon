"""axon 의 백엔드 독립 dtype.

`DType` 은 base class 이고, 각 dtype (`FLOAT16` / `BFLOAT16` / `FLOAT32` / `FLOAT64`
/ `INT32` / `INT64` / `BOOL`) 은 subclass 다. 외부 namespace 에는 `DType` 한 개만
노출되고 하위 클래스들은 모두 class attribute 로 접근한다 (`DType.FLOAT16` 등).

각 멤버가 *클래스* 이므로 generic 자리에 바로 쓸 수 있다:

    class Array[D: DType]: ...

    W: Array[DType.FLOAT32]            # ✓ 클래스이므로 valid type form
    x: Array[DType.INT64] | None       # ✓

Enum 이었다면 `Literal[DType.FLOAT32]` 로 감싸야 했지만, class 기반에서는 불필요.
백엔드 native dtype (np.float32 등) 으로의 변환은 `backend._dtype` 에서 담당.
"""


class DType:
  """모든 axon dtype 의 base class.

  직접 인스턴스화하지 않고, subclass (`DType.FLOAT16` 등) 를 마커처럼 쓴다.
  """

  is_floating: bool = False
  is_integer: bool = False

  # 모듈 하단에서 attach 되는 subclass 들의 forward declaration.
  # 정적 분석기 (basedpyright/pyrefly) 가 `DType.FLOAT16` 을 인식할 수 있게 한다.
  FLOAT16: type[DType]
  BFLOAT16: type[DType]
  FLOAT32: type[DType]
  FLOAT64: type[DType]
  INT32: type[DType]
  INT64: type[DType]
  BOOL: type[DType]


class _FLOAT16(DType):
  is_floating = True


class _BFLOAT16(DType):
  is_floating = True


class _FLOAT32(DType):
  is_floating = True


class _FLOAT64(DType):
  is_floating = True


class _INT32(DType):
  is_integer = True


class _INT64(DType):
  is_integer = True


class _BOOL(DType):
  pass


# 외부 namespace 오염 없이 DType.X 로만 접근하게 attach.
DType.FLOAT16 = _FLOAT16
DType.BFLOAT16 = _BFLOAT16
DType.FLOAT32 = _FLOAT32
DType.FLOAT64 = _FLOAT64
DType.INT32 = _INT32
DType.INT64 = _INT64
DType.BOOL = _BOOL


# ---------------------------------------------------------------------------
# Type promotion
# ---------------------------------------------------------------------------

# rank 가 클수록 더 wide 한 dtype. 같은 비트 폭 중에서는 BFLOAT16 이 FLOAT16 보다
# 다이나믹 레인지가 넓어 한 단계 위에 둠 (NumPy 의 정확한 룰은 아니지만 ML 실용성 기준).
_RANK: dict[type[DType], int] = {
  _BOOL: 0,
  _INT32: 1,
  _INT64: 2,
  _FLOAT16: 3,
  _BFLOAT16: 4,
  _FLOAT32: 5,
  _FLOAT64: 6,
}


def _scalar_dtype(x: object) -> type[DType]:
  """Python scalar → DType 매핑. bool/int/float 외에는 TypeError."""
  if isinstance(x, bool):
    return DType.BOOL
  if isinstance(x, int):
    return DType.INT64
  if isinstance(x, float):
    return DType.FLOAT32
  raise TypeError(f"promote: scalar 가 아님. {type(x).__name__}")


def promote(*items: type[DType] | int | float | bool) -> type[DType]:
  """입력들의 promoted dtype — strong-typing 식 numpy promotion.

  - Python `bool` → `BOOL`, `int` → `INT64`, `float` → `FLOAT32` 매핑
    (axon 의 부동소수 default 가 FLOAT32 라 numpy 의 FLOAT64 와 다름)
  - 결과: 모든 input 중 가장 wide 한 dtype
    `BOOL < INT32 < INT64 < FLOAT16 < BFLOAT16 < FLOAT32 < FLOAT64`

  사용 예:
    `promote(DType.FLOAT32, 2)` → `FLOAT32`     (FLOAT32 vs INT64)
    `promote(DType.INT32, 0.5)` → `FLOAT32`     (INT32 vs FLOAT32)
    `promote(2, 3)` → `INT64`                   (둘 다 Python int)
    `promote(2, 0.5)` → `FLOAT32`               (int vs float)
  """
  dtypes = [
    x if isinstance(x, type) and issubclass(x, DType) else _scalar_dtype(x)
    for x in items
  ]
  return max(dtypes, key=lambda d: _RANK[d])
