from HyperML.optim.riemannian_optim import RiemannianOptimizer
from torch.optim.optimizer import Optimizer, required
import torch
class RAMSGrad(RiemannianOptimizer):
    """Radam adapted for Hyperbolic manifold.

    RAdam algorithm proposed in 'RIEMANNIAN ADAPTIVE OPTIMIZATION METHODS'
    adapted for Poincare Ball Model.
    """
    def __init__(self, params, lr=required, beta=(0.9, 0.999), manifold=required, eps=0.9999999):
        super(RAMSGrad, self).__init__(params, lr=lr, manifold=manifold, eps=eps)
        self.manifold = manifold
        self.beta_1, self.beta_2 = beta
        self.tau = None
        self.v = None

    def _optimization_method(self, p, d_p, lr):
        gradient = d_p
        if(self.tau is None or self.v is None):
            self.tau = torch.zeros(gradient.size()).to(p.device)
            self.v = torch.zeros(gradient.size()).to(p.device)
        # we update only weigth that moved
        mask = (gradient.sum(-1) != 0)
        if(mask.sum() == 0):
            print('No examples to update')
            return
        gradient = gradient[mask]
        gradient = self.manifold.euclidean_to_riemannian_grad(p[mask], gradient)
        tau = self.tau[mask]
        v = self.v[mask]
        m = self.beta_1 * tau + (1 - self.beta_1) * gradient
        self.v[mask] =\
            self.beta_2 * v + (1 - self.beta_2) * self.manifold.norm(p[mask], gradient)
        self.v[mask] = torch.cat((v.unsqueeze(0), self.v[mask].unsqueeze(0))).max(0)[0]

        updated_weight = self.manifold.riemannian_exp(p[mask], - lr * m / torch.sqrt(self.v[mask]))
        self.tau[mask] =\
            self.manifold.parallel_transport(from_point=p[mask],
                                             to_point=updated_weight,
                                             vector=m)
        p[mask] = updated_weight
