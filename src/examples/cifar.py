"""axon 으로 CIFAR-10 학습. 3-layer MLP, ~45% test acc (MLP 의 천장)."""

from __future__ import annotations

import os
import pickle
import tarfile
import time
import urllib.request
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

import axon
from axon import Constant, Node, Var, net
from axon.optim import SGD


_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CACHE_DIR = os.path.expanduser("~/.cache/axon")
_TAR_PATH = os.path.join(_CACHE_DIR, "cifar-10-python.tar.gz")
_EXTRACT_DIR = os.path.join(_CACHE_DIR, "cifar-10-batches-py")

CLASSES = (
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
)


def _download_cifar() -> None:
  os.makedirs(_CACHE_DIR, exist_ok=True)
  if os.path.isdir(_EXTRACT_DIR):
    return
  if not os.path.isfile(_TAR_PATH):
    print(f"downloading CIFAR-10 (~170MB) → {_TAR_PATH}")
    urllib.request.urlretrieve(_CIFAR_URL, _TAR_PATH)
  print("extracting ...")
  with tarfile.open(_TAR_PATH) as tar:
    tar.extractall(_CACHE_DIR)


def _load_batch(path: str) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
  with open(path, "rb") as f:
    d = pickle.load(f, encoding="bytes")
  return d[b"data"], np.array(d[b"labels"], dtype=np.int64)


def load_cifar10() -> tuple[
  NDArray[np.float32], NDArray[np.int64],
  NDArray[np.float32], NDArray[np.int64],
]:
  _download_cifar()
  xs, ts = [], []
  for i in range(1, 6):
    x, t = _load_batch(os.path.join(_EXTRACT_DIR, f"data_batch_{i}"))
    xs.append(x)
    ts.append(t)
  X_train = np.concatenate(xs).astype(np.float32) / 255.0
  T_train = np.concatenate(ts)
  X_test_u8, T_test = _load_batch(os.path.join(_EXTRACT_DIR, "test_batch"))
  X_test = X_test_u8.astype(np.float32) / 255.0
  return X_train, T_train, X_test, T_test


BATCH_SIZE = 128
HIDDEN1 = 256
HIDDEN2 = 128
EPOCHS = 20
LR = 0.1


class MLP(net.Net):
  optimizer = SGD(lr=LR)

  def __init__(self):
    self.fc1 = net.Linear(3072, HIDDEN1, bias=False)
    self.act1 = net.Sigmoid()
    self.fc2 = net.Linear(HIDDEN1, HIDDEN2, bias=False)
    self.act2 = net.Sigmoid()
    self.fc3 = net.Linear(HIDDEN2, 10, bias=False)

  def forward(self, x: Node) -> Var:
    x = self.fc1.forward(x)
    x = self.act1.forward(x)
    x = self.fc2.forward(x)
    x = self.act2.forward(x)
    return self.fc3.forward(x)


def iterate_batches(
  x: NDArray[np.float32],
  t: NDArray[np.int64],
  batch_size: int,
  *,
  shuffle: bool,
) -> Iterator[tuple[NDArray[np.float32], NDArray[np.int64]]]:
  n = x.shape[0]
  idx = np.random.permutation(n) if shuffle else np.arange(n)
  for start in range(0, n, batch_size):
    b = idx[start : start + batch_size]
    yield x[b], t[b]


def main() -> None:
  axon.set_backend("numpy")
  np.random.seed(0)

  print("loading CIFAR-10 (cached) ...")
  X_train, T_train, X_test, T_test = load_cifar10()
  print(f"  X_train {X_train.shape}, T_train {T_train.shape}")
  print(f"  X_test  {X_test.shape}, T_test  {T_test.shape}")

  model = MLP()
  loss_fn = net.CrossEntropyLoss()

  n_params = sum(p._data.size for p in model.parameters())
  print(
    f"params: {n_params:,}, lr={LR}, hidden=({HIDDEN1}, {HIDDEN2}), batch={BATCH_SIZE}"
  )

  train_steps = (X_train.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
  test_steps = (X_test.shape[0] + 256 - 1) // 256

  for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    train_loss = 0.0
    train_correct = 0
    for x_b, t_b in tqdm(
      iterate_batches(X_train, T_train, BATCH_SIZE, shuffle=True),
      total=train_steps,
      leave=False,
    ):
      logits = model.forward(Constant(x_b))
      loss = loss_fn.forward(logits, t_b)
      loss.backward()
      loss.optimize()

      train_loss += float(loss.as_numpy())
      train_correct += int((np.argmax(logits.as_numpy(), axis=1) == t_b).sum())

    test_loss = 0.0
    test_correct = 0
    for x_b, t_b in tqdm(
      iterate_batches(X_test, T_test, 256, shuffle=False),
      total=test_steps,
      leave=False,
    ):
      logits = model.forward(Constant(x_b))
      loss = loss_fn.forward(logits, t_b)
      test_loss += float(loss.as_numpy())
      test_correct += int((np.argmax(logits.as_numpy(), axis=1) == t_b).sum())

    print(
      f"epoch {epoch:2d}/{EPOCHS} | "
      f"train loss={train_loss / train_steps:.4f} "
      f"acc={train_correct / X_train.shape[0]:.4f} | "
      f"test loss={test_loss / test_steps:.4f} "
      f"acc={test_correct / X_test.shape[0]:.4f} | "
      f"{time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
  main()
