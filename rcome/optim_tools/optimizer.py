''' optimisation functions, those methods are deprecated in favor of ramsgrad.py, riemannian_optim.py, rsgd.py
'''

import math

import torch
from torch.optim.optimizer import Optimizer, required

from rcome.function_tools import poincare_function as pf

@DeprecationWarning
class PoincareOptimizer(Optimizer):
    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(PoincareOptimizer, self).__init__(params, defaults)
        self.eps = 1-1e-8

    def __setstate__(self, state):
        super(PoincareOptimizer, self).__setstate__(state)

    def _optimization_method(self, p, d_p, lr):
        return p.new(p.size()).zero_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:  
                if p.grad is None:
                    continue
                d_p = p.grad.data
                self._optimization_method(p.data, d_p.data, lr=group['lr'])

@DeprecationWarning
class PoincareBallSGD(PoincareOptimizer):

    def __init__(self, params, lr=required):
        super(PoincareBallSGD, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        retractation_factor = ((1 - torch.sum(p** 2, dim=-1))**2)/4
        p.add_(-lr, d_p * retractation_factor.unsqueeze(-1).expand_as(d_p))

@DeprecationWarning
class PoincareBallSGDAdd(PoincareOptimizer):
    def __init__(self, params, lr=required):
        super(PoincareBallSGDAdd, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        p.copy_(pf.add(pf.renorm_projection(p.data), -lr*pf.exp(d_p.new(d_p.size()).zero_(),d_p)))

@DeprecationWarning
class PoincareBallSGDExp(PoincareOptimizer):
    def __init__(self, params, lr=required):
        super(PoincareBallSGDExp, self).__init__(params, lr=lr)
        self.first_over = False
    def _optimization_method(self, p, d_p, lr):
        with torch.no_grad():
            a = pf.exp(p, -lr*d_p)

            if(((a.norm(2,-1))>=self.eps).max()>0):
                if(not self.first_over):
                    print("New update out of the disc:", a.norm(2,-1).max())
                    self.first_over = True
                mask = a[a.norm(2,-1)>=self.eps]
                a[a.norm(2,-1)>=self.eps] = p[a.norm(2,-1)>=self.eps] 
            p.copy_(a)

@DeprecationWarning
class PoincareBallSGAExp(PoincareOptimizer):
    def __init__(self, params, lr=required):
        super(PoincareBallSGAExp, self).__init__(params, lr=lr)

    def _optimization_method(self, p, d_p, lr):
        p.copy_(pf.exp(p, lr*d_p))

@DeprecationWarning
class PoincareRAdam(PoincareOptimizer):
    """Radam adapted for Hyperbolic manifold.

    RAdam algorithm proposed in 'Rieammanian Adaptative Optimization Methods'
    adapted for Poincare Ball Model.
    """
    def __init__(self, params, lr=required, beta=(0.9, 0.999)):
        super(PoincareRAdam, self).__init__(params, lr=lr)

        self.beta_1, self.beta_2 = beta
        self.tau = None
        self.v = None

    def _optimization_method(self, p, d_p, lr):
        with torch.no_grad():
            gradient = d_p 

            if(self.tau is None or self.v is None):
                self.tau = torch.zeros(gradient.size()).to(p.device)
                self.v = torch.zeros(gradient.size()).to(p.device)
            # we update only weigth that moved
            mask = (gradient.norm(2, -1) != 0)
            if(mask.sum() == 0):
                return
            gradient = gradient[mask]
            tau = self.tau[mask]
            v = self.v[mask]
            m = self.beta_1 * tau + (1 - self.beta_1) * gradient
            self.v[mask] =\
                self.beta_2 * v + (1 - self.beta_2) * pf.norm(p[mask], gradient)
            self.v[mask] = torch.cat((v.unsqueeze(0), self.v[mask].unsqueeze(0))).max(0)[0]
  
            updated_weight = pf.exp(p[mask], - lr * m / torch.sqrt(self.v[mask]))
            self.tau[mask] =\
                pf.parallel_transport(from_point=p[mask],
                                      to_point=updated_weight,
                                      vector=m)
            p[mask] = updated_weight
