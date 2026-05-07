"""백엔드 독립 dtype. `DType.FLOAT32` 처럼 class attribute 로 접근."""


class DType:
  is_floating: bool = False
  is_integer: bool = False

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


DType.FLOAT16 = _FLOAT16
DType.BFLOAT16 = _BFLOAT16
DType.FLOAT32 = _FLOAT32
DType.FLOAT64 = _FLOAT64
DType.INT32 = _INT32
DType.INT64 = _INT64
DType.BOOL = _BOOL


# rank 클수록 wide. BFLOAT16 이 FLOAT16 보다 다이나믹 레인지 넓어 한 단계 위.
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
  if isinstance(x, bool):
    return DType.BOOL
  if isinstance(x, int):
    return DType.INT64
  if isinstance(x, float):
    return DType.FLOAT32
  raise TypeError(f"promote: scalar 가 아님. {type(x).__name__}")


def promote(*items: type[DType] | int | float | bool) -> type[DType]:
  """numpy-style promotion. Python int → INT64, float → FLOAT32 매핑."""
  dtypes = [
    x if isinstance(x, type) and issubclass(x, DType) else _scalar_dtype(x)
    for x in items
  ]
  return max(dtypes, key=lambda d: _RANK[d])
