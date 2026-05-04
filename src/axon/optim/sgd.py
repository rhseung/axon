from axon.optim.optimizer import Optimizer
from axon.parameter import Parameter


class SGD(Optimizer):
  def optimize(self, p: Parameter):
    p -= self.lr * p.grad
