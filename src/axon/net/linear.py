from axon.net.net import Net
from axon.node import Node
from axon.parameter import Parameter


class Linear(Net):
  def __init__(self, in_features: int, out_features: int):
    self.weight = Parameter((in_features, out_features))
    self.bias = Parameter((out_features,))

  def forward(self, x: Node):
    return x @ self.weight + self.bias
