"""수치 안정성을 위해 primitive 로 둔 Op 들. 자세한 이유는 각 Op docstring."""

from axon.operation.stable.div import Div

__all__ = ["Div"]
