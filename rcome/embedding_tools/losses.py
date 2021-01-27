import torch 
from torch.nn import functional as tf

from rcome.function_tools import poincare_function as pf
from rcome.function_tools import distribution_function as df
from rcome.manifold.poincare_ball import PoincareBallExact


def tree_embedding_criterion(x, y, z, manifold=PoincareBallExact):
    distance_x_y = manifold.distance(x, y)
    distance_x_z = manifold.distance(x.unsqueeze(1).expand_as(z), z)
    loss = distance_x_y + torch.log(torch.exp(-distance_x_y) + torch.exp(-distance_x_z).sum(-1))
    return loss


def graph_embedding_criterion(x, y, z=None, manifold=PoincareBallExact):
    distance_x_y = manifold.distance(x, y)**2
    loss = tf.logsigmoid(-distance_x_y)
    if(z is not None):
        distance_x_z = manifold.distance(x.unsqueeze(1).expand_as(z), z)**2
        loss += tf.logsigmoid(distance_x_z).sum(-1) 
    return -loss


def graph_community_criterion(x, pi, mu, sigma, normalisation_factor, manifold=PoincareBallExact):
    B, M, D = (x.shape[0],) +  mu.shape
    zeta_v = normalisation_factor.unsqueeze(0).expand(B, M)
    x_r = x.unsqueeze(1).expand(B,M,D)
    mu_r = mu.unsqueeze(0).expand(B,M,D)
    sigma_r = sigma.unsqueeze(0).expand(B, M)

    u_pdf = -(manifold.distance(x_r, mu_r)**2)/(2 * sigma_r**2)
    n_pdf = pi.squeeze() * (u_pdf - torch.log(zeta_v))

    return -n_pdf.sum(-1)


@DeprecationWarning
def O1(x, y, manifold, coef=1.):
    manifold.distance


@DeprecationWarning
class SGALoss(object):
    @staticmethod
    def O1(x, y, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        return tf.logsigmoid(-torch.clamp(((distance(x, y) * coef)**2), max=2000))

    @staticmethod
    def O2(x, y, z, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        x_reshape = x.unsqueeze(-2).expand_as(z) * 1
        return SGALoss.O1(x, y, distance=distance, coef=coef) + tf.logsigmoid((distance(x_reshape,z)*coef)**2).sum(-1)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, normalisation_factor, distance=None):
        if(distance is None):
            distance = pf.distance
        B, M, D = (x.shape[0],) +  mu.shape

        # computing normalisation factor

        zeta_v = normalisation_factor.unsqueeze(0).expand(B, M)
        # computing unormalised pdf
        x_r = x.unsqueeze(1).expand(B,M,D)
        mu_r = mu.unsqueeze(0).expand(B,M,D)

        sigma_r = sigma.unsqueeze(0).expand(B, M)
        u_pdf = torch.exp(-(distance(x_r, mu_r)**2)/(2 * sigma_r**2)).clamp(min=1e-10) 
        # normalize the pdf
        n_pdf = pi.squeeze() * torch.log((u_pdf/zeta_v))
        # return the sum over gaussian component 
        return n_pdf.sum(-1)


    @staticmethod
    def O3_fast(x, pi, mu, sigma, normalisation_factor, distance=None):
        if(distance is None):
            distance = pf.distance
        B, M, D = (x.shape[0],) +  mu.shape

        # computing normalisation factor

        zeta_v = normalisation_factor.unsqueeze(0).expand(B, M)
        # computing unormalised pdf
        x_r = x.unsqueeze(1).expand(B,M,D)
        mu_r = mu.unsqueeze(0).expand(B,M,D)

        sigma_r = sigma.unsqueeze(0).expand(B, M)
        u_pdf = -((distance(x_r, mu_r)**2)/(2 * sigma_r**2).clamp(min=1e-10))
        # normalize the pdf
        n_pdf = pi.squeeze() * (u_pdf - torch.log(zeta_v))
        # return the sum over gaussian component 
        return n_pdf.mean(-1)

@DeprecationWarning
class SGDLoss(object):
    @staticmethod
    def O1(x, y, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O1(x, y, distance=distance, coef=coef)

    @staticmethod
    def O2(x, y, z, distance=None, coef=1.):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O2(x, y, z, distance=distance, coef=coef)
    
    # x = BxD
    # pi = BxM
    # mu = BxMxD
    # sigma = BxM
    @staticmethod
    def O3(x, pi, mu, sigma, normalisation_factor, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O3_fast(x, pi.detach(), mu.detach(), sigma.detach(), normalisation_factor.detach(), distance=distance)

@DeprecationWarning
class SGDSoftmaxLoss(object):
    @staticmethod
    def O1(x, y, distance=None):
        if(distance is None):
            distance = pf.distance
        return distance(x, y)**2
    
    @staticmethod
    def O2(x, y, z, distance=None):
        if(distance is None):
            distance = pf.distance
        y_reshape = y.unsqueeze(2).expand_as(z)
        return SGDSoftmaxLoss.O1(x, y) + ( torch.log((-distance(y_reshape,z)**2).exp() )).sum(-1)
    