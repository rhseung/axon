"""NumPy ↔ MLX parity 검증.

각 BackendProtocol 메서드에 대해 동일 입력 → 동일 결과 (atol=1e-5).
CuPy 는 macOS 에서 검증 불가 — `pytest -m cuda` 로 분리.

테스트 분류:
- 생성: zeros/ones/full/eye/arange/*_like
- 산술: 기본 4칙 + matmul (broadcast 포함)
- 수학 함수: exp/log/sqrt/rsqrt/abs/sign/clip/sin/cos/tanh/maximum/minimum/power
- 축소: sum/mean/max/min/var/norm (multi-axis, keepdims, ddof)
- 형상: reshape/transpose/expand_dims/squeeze/broadcast_to/concatenate/stack/split/flip
- 인덱싱: where/take/tril/triu, 슬라이싱, fancy indexing
- 선형대수: matmul (batched), einsum
- 정렬: sort/argsort
- 패딩: pad
- 변환: from_numpy round-trip
- in-place / setitem (backward grad 시나리오)
- DType 매트릭스: 지원/미지원 케이스
- Edge case: 0-d 스칼라, 빈 배열, 음수, 큰 음수
- Random: 통계적 / 결정적 시드
- xp 프록시 동작
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import axon
from axon import DType
from axon.backend import BackendName, BackendProtocol

pytest.importorskip("mlx", reason="mlx 미설치 — macOS 에서만 의미 있음")


# 콜백 파라미터를 BackendProtocol 로 타이핑 → 람다 인자 `xp` 가 BackendProtocol 로
# 추론되어 IDE / pyrefly 자동완성이 살아난다.
Case = Callable[[BackendProtocol], Any]


@pytest.fixture(autouse=True)
def _restore_backend():
  prev = axon.get_backend()
  yield
  axon.set_backend(prev)


def _run(backend: BackendName, fn: Case):
  axon.set_backend(backend)
  return axon.current().to_numpy(fn(axon.xp))


def _both(fn: Case, atol: float = 1e-5, rtol: float = 1e-5):
  np_out = _run("numpy", fn)
  mlx_out = _run("mlx", fn)
  np.testing.assert_allclose(np_out, mlx_out, atol=atol, rtol=rtol)


# =============================================================================
# 생성
# =============================================================================


def test_zeros_ones_full():
  _both(lambda xp: xp.zeros((3, 4), dtype=DType.FLOAT32))
  _both(lambda xp: xp.zeros((), dtype=DType.FLOAT32))  # 0-d
  _both(lambda xp: xp.zeros((0,), dtype=DType.FLOAT32))  # 빈 1-d
  _both(lambda xp: xp.ones((2, 5), dtype=DType.FLOAT32))
  _both(lambda xp: xp.ones((1, 1, 1), dtype=DType.FLOAT32))
  _both(lambda xp: xp.full((3, 3), 2.5, dtype=DType.FLOAT32))
  _both(lambda xp: xp.full((4,), -1.0, dtype=DType.FLOAT32))


def test_zeros_ones_full_like():
  _both(lambda xp: xp.zeros_like(xp.array([[1.0, 2.0], [3.0, 4.0]])))
  _both(lambda xp: xp.ones_like(xp.array([1.0, 2.0, 3.0])))
  _both(lambda xp: xp.full_like(xp.array([[1.0, 2.0], [3.0, 4.0]]), 7.0))


def test_eye():
  _both(lambda xp: xp.eye(4, dtype=DType.FLOAT32))
  _both(lambda xp: xp.eye(1, dtype=DType.FLOAT32))
  _both(lambda xp: xp.eye(5, dtype=DType.FLOAT32))


def test_arange():
  _both(lambda xp: xp.arange(10, dtype=DType.FLOAT32))
  _both(lambda xp: xp.arange(0.0, 5.0, 0.5, dtype=DType.FLOAT32))
  _both(lambda xp: xp.arange(-3.0, 3.0, dtype=DType.FLOAT32))
  _both(lambda xp: xp.arange(0, 10, dtype=DType.INT32))


# =============================================================================
# 산술 (broadcasting 포함)
# =============================================================================


def test_arithmetic_basic():
  data_a = [[1.0, 2.0], [3.0, 4.0]]
  data_b = [[2.0, 1.0], [0.5, 3.0]]
  _both(lambda xp: xp.array(data_a) + xp.array(data_b))
  _both(lambda xp: xp.array(data_a) - xp.array(data_b))
  _both(lambda xp: xp.array(data_a) * xp.array(data_b))
  _both(lambda xp: xp.array(data_a) / xp.array(data_b))
  _both(lambda xp: -xp.array(data_a))


def test_arithmetic_broadcasting():
  # (3, 1) + (1, 4) → (3, 4)
  _both(lambda xp: xp.ones((3, 1)) + xp.ones((1, 4)) * 2.0)
  # (4,) + (3, 4) → (3, 4)
  _both(lambda xp: xp.array([1.0, 2.0, 3.0, 4.0]) + xp.ones((3, 4)))
  # 스칼라 + 배열
  _both(lambda xp: xp.array([1.0, 2.0, 3.0]) * 5.0)
  _both(lambda xp: 3.0 + xp.array([1.0, 2.0, 3.0]))
  _both(lambda xp: 10.0 - xp.array([1.0, 2.0, 3.0]))


def test_matmul_basic():
  a = [[1.0, 2.0], [3.0, 4.0]]
  b = [[2.0, 1.0], [0.5, 3.0]]
  _both(lambda xp: xp.matmul(xp.array(a), xp.array(b)))


def test_matmul_batched():
  np.random.seed(0)
  a = np.random.randn(3, 4, 5).astype(np.float32)
  b = np.random.randn(3, 5, 6).astype(np.float32)

  def fn(xp):
    return xp.matmul(xp.from_numpy(a), xp.from_numpy(b))

  _both(fn, atol=1e-4)


# =============================================================================
# 수학 함수
# =============================================================================


def test_math_unary_positive():
  data = [[0.5, 1.0], [1.5, 2.0]]
  _both(lambda xp: xp.exp(xp.array(data)))
  _both(lambda xp: xp.log(xp.array(data)))
  _both(lambda xp: xp.sqrt(xp.array(data)))
  _both(lambda xp: xp.rsqrt(xp.array(data)))


def test_math_signed():
  data = [[-1.0, 2.0], [3.0, -4.0]]
  _both(lambda xp: xp.abs(xp.array(data)))
  _both(lambda xp: xp.sign(xp.array(data)))
  _both(lambda xp: xp.sign(xp.array([0.0, -0.0, 1.0, -1.0])))


def test_math_trig():
  data = [0.0, 0.5, 1.0, 1.57, 3.14]
  _both(lambda xp: xp.sin(xp.array(data)), atol=1e-6)
  _both(lambda xp: xp.cos(xp.array(data)), atol=1e-6)
  _both(lambda xp: xp.tanh(xp.array([-2.0, -1.0, 0.0, 1.0, 2.0])))


def test_math_clip():
  _both(lambda xp: xp.clip(xp.array([-3.0, -1.0, 0.5, 2.0, 5.0]), -1.0, 2.0))


def test_math_min_max():
  a = [[1.0, 5.0], [3.0, 2.0]]
  b = [[2.0, 1.0], [4.0, 4.0]]
  _both(lambda xp: xp.maximum(xp.array(a), xp.array(b)))
  _both(lambda xp: xp.minimum(xp.array(a), xp.array(b)))


def test_math_power():
  _both(lambda xp: xp.power(xp.array([1.0, 2.0, 3.0]), xp.array([2.0, 2.0, 2.0])))
  _both(lambda xp: xp.power(xp.array([1.0, 2.0, 3.0]), 3.0))


# =============================================================================
# 축소
# =============================================================================


def test_reductions_simple():
  data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  _both(lambda xp: xp.sum(xp.array(data)))
  _both(lambda xp: xp.sum(xp.array(data), axis=0))
  _both(lambda xp: xp.sum(xp.array(data), axis=1))
  _both(lambda xp: xp.mean(xp.array(data), axis=0))
  _both(lambda xp: xp.mean(xp.array(data), axis=1))
  _both(lambda xp: xp.max(xp.array(data), axis=0))
  _both(lambda xp: xp.min(xp.array(data), axis=1))


def test_reductions_keepdims():
  data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  _both(lambda xp: xp.sum(xp.array(data), axis=1, keepdims=True))
  _both(lambda xp: xp.mean(xp.array(data), axis=0, keepdims=True))
  _both(lambda xp: xp.max(xp.array(data), keepdims=True))


def test_reductions_var():
  data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  _both(lambda xp: xp.var(xp.array(data)))
  _both(lambda xp: xp.var(xp.array(data), axis=1))
  _both(lambda xp: xp.var(xp.array(data), axis=1, ddof=1))
  _both(lambda xp: xp.var(xp.array(data), axis=0, keepdims=True))


def test_reductions_3d():
  np.random.seed(1)
  data = np.random.randn(2, 3, 4).astype(np.float32)
  _both(lambda xp: xp.sum(xp.from_numpy(data), axis=0), atol=1e-4)
  _both(lambda xp: xp.sum(xp.from_numpy(data), axis=1), atol=1e-4)
  _both(lambda xp: xp.sum(xp.from_numpy(data), axis=2), atol=1e-4)
  _both(lambda xp: xp.mean(xp.from_numpy(data), axis=2, keepdims=True), atol=1e-5)


def test_norm():
  _both(lambda xp: xp.norm(xp.array([3.0, 4.0])))
  _both(lambda xp: xp.norm(xp.array([3.0, 4.0]), ord=2))
  _both(lambda xp: xp.norm(xp.array([1.0, -2.0, 3.0]), ord=1))
  _both(lambda xp: xp.norm(xp.array([[1.0, 2.0], [3.0, 4.0]]), axis=1))
  _both(lambda xp: xp.norm(xp.array([[1.0, 2.0], [3.0, 4.0]]), axis=0))


# =============================================================================
# 형상
# =============================================================================


def test_reshape():
  data = list(range(12))
  _both(lambda xp: xp.reshape(xp.array(data, dtype=DType.FLOAT32), (3, 4)))
  _both(lambda xp: xp.reshape(xp.array(data, dtype=DType.FLOAT32), (2, 2, 3)))
  _both(lambda xp: xp.reshape(xp.array(data, dtype=DType.FLOAT32), (12,)))


def test_transpose():
  np.random.seed(2)
  data = np.random.randn(2, 3, 4).astype(np.float32)
  _both(lambda xp: xp.transpose(xp.from_numpy(data)))  # 기본: 모든 축 역순
  _both(lambda xp: xp.transpose(xp.from_numpy(data), axes=(1, 0, 2)))
  _both(lambda xp: xp.transpose(xp.from_numpy(data), axes=(2, 1, 0)))


def test_expand_squeeze():
  _both(lambda xp: xp.expand_dims(xp.array([1.0, 2.0, 3.0]), 0))
  _both(lambda xp: xp.expand_dims(xp.array([1.0, 2.0, 3.0]), 1))
  _both(lambda xp: xp.expand_dims(xp.array([1.0, 2.0, 3.0]), -1))
  _both(lambda xp: xp.squeeze(xp.array([[[1.0]], [[2.0]]])))
  _both(lambda xp: xp.squeeze(xp.array([[[1.0, 2.0, 3.0]]]), axis=0))


def test_broadcast_to():
  _both(lambda xp: xp.broadcast_to(xp.array([1.0, 2.0, 3.0]), (4, 3)))
  _both(lambda xp: xp.broadcast_to(xp.array([[1.0], [2.0]]), (2, 5)))


def test_concat_stack():
  _both(
    lambda xp: xp.concatenate([xp.array([[1.0, 2.0]]), xp.array([[3.0, 4.0]])], axis=0)
  )
  _both(
    lambda xp: xp.concatenate(
      [xp.array([[1.0], [2.0]]), xp.array([[3.0], [4.0]])], axis=1
    )
  )
  _both(lambda xp: xp.stack([xp.array([1.0, 2.0]), xp.array([3.0, 4.0])], axis=0))
  _both(lambda xp: xp.stack([xp.array([1.0, 2.0]), xp.array([3.0, 4.0])], axis=1))


def test_flip():
  data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  _both(lambda xp: xp.flip(xp.array(data), axis=0))
  _both(lambda xp: xp.flip(xp.array(data), axis=1))
  _both(lambda xp: xp.flip(xp.array(data)))  # 모든 축


def test_split():
  def split_int(xp):
    parts = xp.split(xp.array([1.0, 2.0, 3.0, 4.0]), 2)
    return xp.stack(parts, axis=0)

  def split_list(xp):
    parts = xp.split(xp.array([1.0, 2.0, 3.0, 4.0, 5.0]), [2, 4])
    return xp.concatenate(parts, axis=0)

  def split_axis(xp):
    parts = xp.split(xp.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4)), 2, axis=1)
    return xp.stack(parts, axis=0)

  _both(split_int)
  _both(split_list)
  _both(split_axis)


# =============================================================================
# 인덱싱
# =============================================================================


def test_where():
  _both(
    lambda xp: xp.where(
      xp.array([[True, False], [False, True]]),
      xp.array([[1.0, 2.0], [3.0, 4.0]]),
      xp.array([[10.0, 20.0], [30.0, 40.0]]),
    )
  )


def test_take():
  data = [10.0, 20.0, 30.0, 40.0, 50.0]
  _both(lambda xp: xp.take(xp.array(data), xp.array([0, 2, 4], dtype=DType.INT32)))
  _both(lambda xp: xp.take(xp.array(data), xp.array([4, 2, 0, 1], dtype=DType.INT32)))
  # 2D
  _both(
    lambda xp: xp.take(
      xp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
      xp.array([0, 2], dtype=DType.INT32),
      axis=1,
    )
  )


def test_tril_triu():
  _both(lambda xp: xp.tril(xp.ones((4, 4), dtype=DType.FLOAT32)))
  _both(lambda xp: xp.tril(xp.ones((4, 4), dtype=DType.FLOAT32), k=1))
  _both(lambda xp: xp.tril(xp.ones((4, 4), dtype=DType.FLOAT32), k=-1))
  _both(lambda xp: xp.triu(xp.ones((4, 4), dtype=DType.FLOAT32)))
  _both(lambda xp: xp.triu(xp.ones((4, 4), dtype=DType.FLOAT32), k=1))


def test_slicing():
  data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
  _both(lambda xp: xp.array(data)[0])
  _both(lambda xp: xp.array(data)[:, 0])
  _both(lambda xp: xp.array(data)[1:, 1:])
  _both(lambda xp: xp.array(data)[::-1])
  _both(lambda xp: xp.array(data)[..., 0])
  _both(lambda xp: xp.array(data)[-1])
  _both(lambda xp: xp.array(data)[:, -1])


def test_fancy_indexing_2d():
  """2D fancy indexing — Embedding/CrossEntropy 의 정답 픽업 패턴.

  arr[batch_idx, col_idx] 형태가 양 백엔드에서 동일 동작해야 함.
  """
  data = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]

  def fn(xp):
    arr = xp.array(data)
    batch_idx = xp.array([0, 1, 0], dtype=DType.INT32)
    col_idx = xp.array([1, 2, 0], dtype=DType.INT32)
    return arr[batch_idx, col_idx]

  _both(fn)


def test_fancy_indexing_ce_pattern():
  """CrossEntropy 가 쓰는 패턴: logits[arange(N), targets]."""
  np.random.seed(8)
  logits_np = np.random.randn(4, 10).astype(np.float32)
  targets_np = np.array([3, 0, 7, 2], dtype=np.int32)

  def fn(xp):
    logits = xp.from_numpy(logits_np)
    targets = xp.from_numpy(targets_np)
    n = xp.arange(4, dtype=DType.INT32)
    return logits[n, targets]

  _both(fn)


def test_bool_mask_via_where():
  """MLX 는 arr[mask] 를 미지원하므로 axon 의 권장 패턴은 xp.where 사용.

  filtering (가변 길이) 은 MLX 가 지원 안 함 — 이 패턴은 axon 에서 권장하지 않음.
  대신 mask 적용 (zero-out / replace) 은 양쪽 다 지원.
  """
  _both(
    lambda xp: xp.where(
      xp.array([True, False, True, False, True]),
      xp.array([1.0, 2.0, 3.0, 4.0, 5.0]),
      xp.zeros((5,), dtype=DType.FLOAT32),
    )
  )
  # attention 마스킹 패턴 — causal mask + -inf
  _both(
    lambda xp: xp.where(
      xp.tril(xp.ones((4, 4), dtype=DType.FLOAT32)) > 0,
      xp.from_numpy(np.arange(16, dtype=np.float32).reshape(4, 4)),
      xp.full((4, 4), -1e9, dtype=DType.FLOAT32),
    )
  )


def test_mlx_bool_mask_filtering_unsupported():
  """문서화: MLX 는 bool mask 로 filtering (arr[mask]) 을 지원 안 함.

  명시 테스트로 박아둬서 향후 MLX 가 지원 시 알 수 있게 함. 정확한 메시지는
  버전마다 다름 ("boolean indices are not yet supported" / "indices must be integral")
  이라 패턴은 느슨하게.
  """
  axon.set_backend("mlx")
  arr = axon.xp.array([1.0, 2.0, 3.0, 4.0, 5.0])
  mask = axon.xp.array([True, False, True, False, True])
  with pytest.raises((ValueError, TypeError)):
    _ = arr[mask]


# =============================================================================
# 선형대수
# =============================================================================


def test_einsum_matmul():
  np.random.seed(3)
  a = np.random.randn(2, 3, 4).astype(np.float32)
  b = np.random.randn(2, 4, 5).astype(np.float32)

  _both(
    lambda xp: xp.einsum("bij,bjk->bik", xp.from_numpy(a), xp.from_numpy(b)),
    atol=1e-4,
  )


def test_einsum_attention_shape():
  """attention 형태 — Q @ K^T, output 형태 검증."""
  np.random.seed(4)
  q = np.random.randn(2, 3, 4, 8).astype(np.float32)  # (B, H, T, D)
  k = np.random.randn(2, 3, 4, 8).astype(np.float32)

  _both(
    lambda xp: xp.einsum("bhid,bhjd->bhij", xp.from_numpy(q), xp.from_numpy(k)),
    atol=1e-4,
  )


def test_einsum_transpose():
  np.random.seed(5)
  a = np.random.randn(3, 4).astype(np.float32)
  _both(lambda xp: xp.einsum("ij->ji", xp.from_numpy(a)))


def test_einsum_sum():
  np.random.seed(6)
  a = np.random.randn(3, 4).astype(np.float32)
  _both(lambda xp: xp.einsum("ij->", xp.from_numpy(a)), atol=1e-4)


# =============================================================================
# 정렬
# =============================================================================


def test_sort_argsort():
  data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
  _both(lambda xp: xp.sort(xp.array(data)))
  _both(lambda xp: xp.argsort(xp.array(data)))
  data_2d = [[3.0, 1.0, 4.0], [9.0, 2.0, 6.0]]
  _both(lambda xp: xp.sort(xp.array(data_2d), axis=1))
  _both(lambda xp: xp.sort(xp.array(data_2d), axis=0))


def test_argmax_argmin():
  """argmax/argmin — 값은 양 백엔드 동일. dtype 은 다름 (numpy=int64, mlx=uint32)."""
  data_1d = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
  _both(lambda xp: xp.argmax(xp.array(data_1d)))
  _both(lambda xp: xp.argmin(xp.array(data_1d)))
  data_2d = [[3.0, 1.0, 4.0], [9.0, 2.0, 6.0]]
  _both(lambda xp: xp.argmax(xp.array(data_2d), axis=0))
  _both(lambda xp: xp.argmax(xp.array(data_2d), axis=1))
  _both(lambda xp: xp.argmin(xp.array(data_2d), axis=1))
  _both(lambda xp: xp.argmax(xp.array(data_2d), axis=1, keepdims=True))


def test_cumsum():
  _both(lambda xp: xp.cumsum(xp.array([1.0, 2.0, 3.0, 4.0])))
  _both(lambda xp: xp.cumsum(xp.array([[1.0, 2.0], [3.0, 4.0]]), axis=0))
  _both(lambda xp: xp.cumsum(xp.array([[1.0, 2.0], [3.0, 4.0]]), axis=1))
  # axis=None → flatten 후 누적 (numpy 동작)
  _both(lambda xp: xp.cumsum(xp.array([[1.0, 2.0], [3.0, 4.0]])))


# =============================================================================
# 패딩
# =============================================================================


def test_pad():
  _both(
    lambda xp: xp.pad(
      xp.array([[1.0, 2.0], [3.0, 4.0]]),
      ((1, 1), (1, 1)),
      constant_values=0.0,
    )
  )
  _both(
    lambda xp: xp.pad(
      xp.array([[1.0, 2.0], [3.0, 4.0]]),
      ((2, 0), (0, 3)),
      constant_values=-1.0,
    )
  )


# =============================================================================
# 변환 round-trip
# =============================================================================


def test_from_numpy_round_trip():
  for backend in ("numpy", "mlx"):
    axon.set_backend(backend)
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr = axon.xp.from_numpy(src)
    out = axon.xp.to_numpy(arr)
    np.testing.assert_array_equal(src, out)


# =============================================================================
# DType 매트릭스
# =============================================================================


def test_mlx_rejects_float64():
  axon.set_backend("mlx")
  with pytest.raises(TypeError, match="FLOAT64"):
    axon.xp.zeros((2, 2), dtype=DType.FLOAT64)


def test_mlx_supports_int64():
  axon.set_backend("mlx")
  arr = axon.xp.zeros((2, 2), dtype=DType.INT64)
  assert arr.shape == (2, 2)


def test_numpy_supports_float64():
  axon.set_backend("numpy")
  arr = axon.xp.zeros((2, 2), dtype=DType.FLOAT64)
  assert arr.dtype == np.float64


@pytest.mark.parametrize(
  "dtype",
  [DType.FLOAT16, DType.FLOAT32, DType.INT32, DType.INT64, DType.BOOL],
)
def test_dtype_creation_both_backends(dtype: type[DType]):
  """공통 dtype 은 두 백엔드 모두에서 생성 가능."""
  for backend in ("numpy", "mlx"):
    axon.set_backend(backend)
    arr = axon.xp.zeros((3, 3), dtype=dtype)
    assert arr.shape == (3, 3)


def test_bfloat16_numpy_falls_back_to_fp32():
  """NumPy 는 native bfloat16 없음 — fp32 로 매핑."""
  axon.set_backend("numpy")
  arr = axon.xp.zeros((2, 2), dtype=DType.BFLOAT16)
  assert arr.dtype == np.float32


def test_bfloat16_mlx_native():
  """MLX 는 native bfloat16 있음."""
  import mlx.core as mx

  axon.set_backend("mlx")
  arr = axon.xp.zeros((2, 2), dtype=DType.BFLOAT16)
  assert arr.dtype == mx.bfloat16


# =============================================================================
# In-place / setitem (backward grad 시나리오)
# =============================================================================


def test_inplace_accumulation():
  """backward() 의 grad 누적이 양 백엔드에서 동일."""

  def fn(xp):
    g = xp.zeros((3, 3), dtype=DType.FLOAT32)
    delta = xp.ones((3, 3), dtype=DType.FLOAT32)
    g += delta
    g += delta * 2.0
    g -= delta
    return g

  _both(fn)


def test_inplace_with_broadcasting():
  """broadcast 후 in-place 가 (NumPy 기준) 가능한 형태인지 검증."""

  def fn(xp):
    g = xp.zeros((3, 3), dtype=DType.FLOAT32)
    g += xp.array([1.0, 2.0, 3.0], dtype=DType.FLOAT32)  # (3,) → (3, 3)
    return g

  _both(fn)


def test_setitem():
  def fn(xp):
    g = xp.zeros((3, 3), dtype=DType.FLOAT32)
    g[0, 0] = 5.0
    g[1] = xp.array([1.0, 2.0, 3.0], dtype=DType.FLOAT32)
    g[:, 2] = -1.0
    return g

  _both(fn)


# =============================================================================
# Edge cases
# =============================================================================


def test_scalar_0d():
  """0-d 스칼라 텐서."""
  _both(lambda xp: xp.array(5.0, dtype=DType.FLOAT32))
  _both(lambda xp: xp.exp(xp.array(1.0, dtype=DType.FLOAT32)))
  _both(lambda xp: xp.array(3.0, dtype=DType.FLOAT32) + xp.array(4.0, dtype=DType.FLOAT32))


def test_negative_zero():
  """음수 0 처리."""
  _both(lambda xp: xp.array([-0.0, 0.0]) * 1.0)
  _both(lambda xp: xp.sign(xp.array([-0.0, 0.0, 1.0, -1.0])))


def test_large_negative():
  """큰 음수 입력에서 exp 가 0 으로 underflow."""
  _both(lambda xp: xp.exp(xp.array([-100.0, -50.0, 0.0])))


def test_high_dim_4d():
  """4D 연산 (Transformer 스타일)."""
  np.random.seed(7)
  data = np.random.randn(2, 3, 4, 5).astype(np.float32)
  _both(lambda xp: xp.sum(xp.from_numpy(data), axis=-1), atol=1e-5)
  _both(lambda xp: xp.transpose(xp.from_numpy(data), axes=(0, 2, 1, 3)))


# =============================================================================
# Random — 결정적 시드 + 통계적 성질
# =============================================================================


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_random_seed_reproducible(backend: BackendName):
  """같은 시드로 같은 백엔드에서 두 번 호출 → 같은 값."""
  axon.set_backend(backend)
  axon.xp.random.seed(42)
  a = axon.current().to_numpy(axon.xp.random.normal((100,)))
  axon.xp.random.seed(42)
  b = axon.current().to_numpy(axon.xp.random.normal((100,)))
  np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_random_normal_stats(backend: BackendName):
  """충분히 큰 샘플의 평균/분산이 이론값에 근접."""
  axon.set_backend(backend)
  axon.xp.random.seed(0)
  arr = axon.current().to_numpy(axon.xp.random.normal((50_000,), mean=2.0, std=3.0))
  assert abs(arr.mean() - 2.0) < 0.1
  assert abs(arr.std() - 3.0) < 0.1


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_random_uniform_bounds(backend: BackendName):
  axon.set_backend(backend)
  axon.xp.random.seed(0)
  arr = axon.current().to_numpy(axon.xp.random.uniform((10_000,), low=-1.0, high=1.0))
  assert arr.min() >= -1.0
  assert arr.max() <= 1.0


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_random_bernoulli(backend: BackendName):
  axon.set_backend(backend)
  axon.xp.random.seed(0)
  arr = axon.current().to_numpy(axon.xp.random.bernoulli(0.3, (10_000,)))
  # 0/1 만 나와야
  unique = set(np.unique(arr).tolist())
  assert unique <= {0.0, 1.0}
  # p=0.3 근처
  assert abs(arr.mean() - 0.3) < 0.05


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_random_dtype(backend: BackendName):
  axon.set_backend(backend)
  axon.xp.random.seed(0)
  arr = axon.xp.random.normal((10,), dtype=DType.FLOAT16)
  out = axon.current().to_numpy(arr)
  assert out.dtype == np.float16


# =============================================================================
# xp 프록시 동작
# =============================================================================


def test_xp_proxy_is_live():
  """set_backend 후에도 같은 xp 변수가 새 백엔드에 위임."""
  from axon import xp

  axon.set_backend("numpy")
  out_np = xp.to_numpy(xp.array([1.0, 2.0]))
  assert isinstance(out_np, np.ndarray)

  axon.set_backend("mlx")
  out_mlx = xp.to_numpy(xp.array([1.0, 2.0]))
  assert isinstance(out_mlx, np.ndarray)

  np.testing.assert_allclose(out_np, out_mlx)


def test_xp_random_proxy():
  """xp.random 도 라이브 프록시."""
  from axon import xp

  axon.set_backend("numpy")
  xp.random.seed(0)
  out_np = axon.current().to_numpy(xp.random.normal((10,)))

  axon.set_backend("mlx")
  xp.random.seed(0)
  _ = axon.current().to_numpy(xp.random.normal((10,)))
  # bitwise 일치는 어려우니 shape 만 검증
  assert out_np.shape == (10,)


def test_get_backend_returns_current():
  axon.set_backend("numpy")
  assert axon.get_backend() == "numpy"
  axon.set_backend("mlx")
  assert axon.get_backend() == "mlx"


def test_set_invalid_backend():
  with pytest.raises(ValueError, match="알 수 없는 백엔드"):
    axon.set_backend("foo")  # type: ignore[arg-type]
