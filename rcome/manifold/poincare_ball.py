from rcome.manifold.manifold import Manifold
from rcome.manifold.tools import gyration
from rcome.function_tools import function

import torch 
import math

class PoincareBall(Manifold):

    @staticmethod
    def _lambda(x, keepdim=False):
        return 2 / (1 - x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(1e-15)

    @staticmethod
    def parallel_transport(from_point, to_point, vector):
        """Implementation of parallel transport for Poincare Ball.

        Transport a vector in the tangent point of from_point
        to the tangent space of to_point.
        """
        transported_gyrovector = gyration(to_point, -from_point, vector, 1.)
        lambda_x = PoincareBall._lambda(from_point, keepdim=True)
        lambda_y = PoincareBall._lambda(to_point, keepdim=True)
        return transported_gyrovector * (lambda_x / lambda_y)

    @staticmethod
    def norm(from_point, point):
        return PoincareBall._lambda(from_point, keepdim=True)**2\
            * (point**2).sum(-1, keepdim=True)

    @staticmethod
    def riemannian_log(x, y):
        """Logarithm map for poincare ball representation

        Parameters
        ----------
        x : Tenor<float>
            point x
        y : Tensor<float>
            point to ptojrct to tangent space

        Returns
        -------
        Tensor<float>
            The projection of y
        """
        xpy = PoincareBall.add(-x, y)
        norm_xpy = xpy.norm(2, -1, keepdim=True).expand_as(xpy)
        norm_x = x.norm(2, -1, keepdim=True).expand_as(xpy)
        res = (1 - norm_x**2) * (function.arc_tanh(norm_xpy)) * (xpy / norm_xpy)

        return res

    @staticmethod
    def riemannian_exp(x, v):
        pass


    @staticmethod
    def add(x, y):
        """ Compute Moebus addition

            Parameters
            ----------
            x : Tensor<float>
                first points (using batch)
            y : Tensor<float>
                second points (using batch)

            Returns
            ------
            Tensor<float>
            Results of Moebus addition
        """
        nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x) * 1
        ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x) * 1
        xy = (x * y).sum(-1, keepdim=True).expand_as(x) * 1
        return ((1 + 2 * xy + ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)

class PoincareBallApproximation(PoincareBall):

    class PoincareDistance(torch.autograd.Function):
        """ Auto-grad object for Poincaré Ball distance

        """
        @staticmethod
        def grad(x, v, sqnormx, sqnormv, sqdist, eps):
            alpha = (1 - sqnormx)
            beta = (1 - sqnormv)
            z = 1 + 2 * sqdist / (alpha * beta)
            a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))\
                .unsqueeze(-1).expand_as(x)
            a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
            z = torch.sqrt(torch.pow(z, 2) - 1)
            z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
            return 4 * a / z.expand_as(x)

        @staticmethod
        def forward(ctx, u, v):
            eps = 1e-5
            squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
            sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
            sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
            ctx.eps = eps
            ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
            x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1

            z = torch.sqrt(torch.pow(x, 2) - 1)
            return torch.log(x + z)

        @staticmethod
        def backward(ctx, g):
            '''Return euclidean grad
            '''
            u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
            g = g.unsqueeze(-1)
            gu = PoincareBallApproximation.PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
            gv = PoincareBallApproximation.PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
            gu, gv = g.expand_as(gu) * gu, g.expand_as(gv) * gv
            return gu, gv

    @staticmethod
    def distance(x, y):
        """ Compute poincare ball distance

            Parameters
            ----------
            x : Tensor<float>
                first points (using batch)
            y : Tensor<float>
                second points (using batch)

            Returns
            ------
            Tensor<float>
            Poincare ball distance
        """
        return PoincareBallApproximation.PoincareDistance.apply(x, y)

    @staticmethod
    def barycenter(z, lr=1e-1, tau=5e-4, max_iter=math.inf):
        with torch.no_grad():
            barycenter = z.mean(0, keepdim=True)
            if(len(z) == 1):
                return z
            iteration = 0
            cvg = math.inf
            while(cvg > tau and max_iter > iteration):
                iteration += 1
                grad_tangent = 2 * PoincareBallApproximation.riemannian_log(barycenter.expand_as(z), z)
                grad_tangent /= len(z)
                cc_barycenter = PoincareBallApproximation.riemannian_exp(barycenter, lr * grad_tangent.sum(0, keepdim=True))
                cvg = PoincareBallApproximation.distance(cc_barycenter, barycenter).max().item()
                barycenter = cc_barycenter
            return barycenter


    @staticmethod
    def riemannian_exp(x, k):
        """Exponential map approximmation from
        "Poincaré Embeddings for Learning Hierarchical Representations" (nickel et al 2018)
        It is not the real exponential map but a retraction (for real exp map use PoincareBallExact)

        Parameters
        ----------
        x : Tensor<float>
            Point to project on the hyperplane generated by 0

        Returns
        -------
        Tensor<float>
            The projection of x on the disk
        """
        exp = x + k
        norm = exp.norm(2, -1)
        outter_disc = norm >= 1.
        if(outter_disc.shape[0] >= 1):
            exp[outter_disc] =\
                torch.einsum('ij,i...->ij', exp[outter_disc], 1 / (norm[outter_disc] + 1e-5))
        return exp

    @staticmethod
    def euclidean_to_riemannian_grad(x, euclidean_grad):
        retracted_gx = ((1 - torch.sum(x**2, dim=-1))**2) / 4
        res_x = torch.einsum('...i,...ij->...ij', retracted_gx, euclidean_grad)
        return res_x



class PoincareBallExact(PoincareBall):

    @staticmethod
    def barycenter(z, lr=1e-1, tau=5e-4, max_iter=math.inf):
        with torch.no_grad():
            barycenter = z.mean(0, keepdim=True)
            if(len(z) == 1):
                return z
            iteration = 0
            cvg = math.inf
            while(cvg > tau and max_iter > iteration):
                iteration += 1
                grad_tangent = 2 * PoincareBallExact.riemannian_log(barycenter.expand_as(z), z)
                grad_tangent /= len(z)
                cc_barycenter = PoincareBallExact.riemannian_exp(barycenter, lr * grad_tangent.sum(0, keepdim=True))
                cvg = PoincareBallExact.distance(cc_barycenter, barycenter).max().item()
                barycenter = cc_barycenter
            return barycenter

    @staticmethod
    def distance(x, y):
        """ Compute poincare ball distance

            Parameters
            ----------
            x : Tensor<float>
                first points (using batch)
            y : Tensor<float>
                second points (using batch)

            Returns
            ------
            Tensor<float>
            Poincare ball distance
        """
        return PoincareBallApproximation.PoincareDistance.apply(x, y)



    @staticmethod
    def riemannian_exp(x, v):
        """Exponential map for poincare ball representation

        Parameters
        ----------
        x : Tensor<float>
            point on the manifold
        v : Tensor<float>
            vector on the tangent space

        Returns
        -------
        Tensor<float>
            The projection of v on the tangent space on x
        """
        norm_x = x.norm(2, -1, keepdim=True).expand_as(x)
        lambda_x = 1 / (1 - norm_x**2)
        norm_v = v.norm(2, -1, keepdim=True).expand_as(v)
        direction = v / norm_v
        factor = torch.tanh(lambda_x * norm_v)
        res = PoincareBall.add(x, direction * factor)
        if(0 != len(torch.nonzero(norm_x == 0))):
            res[norm_v == 0] = x[norm_v == 0]
        return res


    @staticmethod
    def euclidean_to_riemannian_grad(from_point, euclidean_grad):
        """From euclidean gradient compute the riemanian gradiant (tangent space).

        Poincare gradient from euclidean gradient, consist in
        rescaling euclidean gradient according to the metric

        Parameters
        ----------
        from_point : Tensor<float>
            Point lying in the manifold
        euclidean_grad : Tensor<float>
            The euclidean gradient

        Returns
        -------
        Tensor<float>
            The riemannian gradient according to from_point
        """

        retracted_gx = ((1 - torch.sum(from_point ** 2, dim=-1))**2) / 4
        res_x = torch.einsum('...i,...ij->...ij', retracted_gx, euclidean_grad)
        return res_x

def test_log_gradient_distance():
    torch.set_default_tensor_type(torch.DoubleTensor)
    for i in range(10):
        x = torch.randn(1,2)
        if(x.norm(2,-1).sum() >1):
            x /= (x.norm(2,-1).sum() + 1e-3)
        x.requires_grad_(requires_grad=True)

        y = torch.randn(1,2)
        if(y.norm(2,-1).sum() >1):
            y /= (y.norm(2,-1).sum() + 1e-3)
        PoincareBallExact.distance(x,y).backward()
        decomposed_grad = PoincareBallExact.euclidean_to_riemannian_grad(x, x.grad)
        log_grad = -PoincareBallExact.riemannian_log(x, y)/PoincareBallExact.distance(x, y)
        assert((decomposed_grad - log_grad).abs().sum() < 1e-8)

test_log_gradient_distance()