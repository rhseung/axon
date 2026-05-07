"""Loss 모듈 — `axon.functional` fused loss 의 thin wrapper."""

from axon.net.loss.cross_entropy import CrossEntropyLoss
from axon.net.loss.mse import MSELoss

__all__ = ["CrossEntropyLoss", "MSELoss"]
