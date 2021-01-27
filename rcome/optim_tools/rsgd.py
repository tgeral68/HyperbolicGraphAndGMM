
from rcome.optim_tools.riemannian_optim import RiemannianOptimizer
from torch.optim.optimizer import Optimizer, required

class RSGD(RiemannianOptimizer):
    """Radam adapted for Hyperbolic manifold.

    RAdam algorithm proposed in 'RIEMANNIAN ADAPTIVE OPTIMIZATION METHODS'
    adapted for Poincare Ball Model.
    """
    def __init__(self, params, lr=required, manifold=required, eps=0.9999999):
        super(RSGD, self).__init__(params, lr=lr, manifold=manifold, eps=eps)
        self.manifold = manifold

    def _optimization_method(self, p, d_p, lr):
        gradient = self.manifold.euclidean_to_riemannian_grad(p, d_p)
        # we update only weigth that moved
        mask = (gradient.sum(-1) != 0)
        if(mask.sum() == 0):
            # print('No examples to update')
            return
        gradient = gradient[mask]

        p[mask] = self.manifold.riemannian_exp(p[mask], - lr * gradient)
