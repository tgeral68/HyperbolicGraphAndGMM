import math
from rcome.manifold.manifold import Manifold


class Euclidean(Manifold):

    @staticmethod
    def barycenter(z, lr=1e-1, tau=5e-4, max_iter=math.inf):
        return z.mean(0)

    @staticmethod
    def parallel_transport(from_point, to_point, vector):
        """Implementation of parallel transport for Poincare Ball.

        Transport a vector in the tangent point of from_point
        to the tangent space of to_point.
        """
        return vector

    @staticmethod
    def norm(from_point, point):
        return point.norm(2,-1)

    @staticmethod
    def distance(x, y):
        return (x - y).norm(2, -1)

    @staticmethod
    def riemannian_log(x, y):
        return x - y


    @staticmethod
    def riemannian_exp(x, v):
        return x + v
    
    @staticmethod
    def euclidean_to_riemannian_grad(x, dx):
        return dx
