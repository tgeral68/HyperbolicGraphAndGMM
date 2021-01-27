from torch.optim.optimizer import Optimizer, required

import torch

class RiemannianOptimizer(Optimizer):
    '''
        Poincare Ball/Disk generic optimizer

        Args:
            params: parameters of the model using pytorch syntax

        Attributes:
            eps (float): a small value used to approximation to
            avoid precision issue

    '''
    def __init__(self, params, lr=required, manifold=required, eps=0.9999999):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(RiemannianOptimizer, self).__init__(params, defaults)
        self.eps = eps
        self.manifold = manifold

    def __setstate__(self, state):
        super(RiemannianOptimizer, self).__setstate__(state)

    def _optimization_method(self, p, d_p, lr):
        pass

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                self._optimization_method(p.data, d_p.data, lr=group['lr'])