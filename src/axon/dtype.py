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
