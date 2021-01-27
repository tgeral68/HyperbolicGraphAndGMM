import numpy as np
import math
import torch
from torch import nn
from rcome.function_tools import poincare_function as pf

pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)
ZETA_CST = math.sqrt(math.pi/2)


def weighted_gmm_pdf(w, z, mu, sigma, distance, norm_func=None):
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    mu_u = mu.unsqueeze(0).expand_as(z_u)

    distance_to_mean = distance(z_u, mu_u)
    sigma_u = sigma.unsqueeze(0).expand_as(distance_to_mean)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma_u**2))
    if(norm_func is None):
        zeta_sigma = pi_2_3 * sigma *  torch.exp((sigma**2/2) * erf_approx(sigma/math.sqrt(2)))
    else:
        zeta_sigma = norm_func(sigma)
    return w*(distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal).detach())

def surface(hypersphere_dim):
    d = float(hypersphere_dim+1)
    return (2* (math.pi**(d/2)))/(math.gamma(d/2))

def zeta_disc(sigma):
    return  math.sqrt(math.pi/2) * sigma * torch.exp(sigma**2/2) * torch.erf(sigma/math.sqrt(2)) * surface(1) 

def log_grad_zeta(x, N):
    sigma = nn.Parameter(x)
    binomial_coefficient=None
    M = sigma.shape[0]
    sigma_u = sigma.unsqueeze(0).expand(N,M)
    if(binomial_coefficient is None):
        # we compute coeficient
        v = torch.arange(N)
        v[0] = 1
        n_fact = v.prod()
        k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
        nmk_fact = k_fact.flip(0)
        binomial_coefficient = n_fact/(k_fact * nmk_fact)  
    binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).double()
    range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
    ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()

    alternate_neg = (-ones_)**(range_)
    ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
    bs_o = binomial_coefficient * as_o
    r = alternate_neg * bs_o    
    logv = torch.log(surface(N-1) *  ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1))))
    logv.sum().backward()
    log_grad = sigma.grad
    return sigma.grad.data



def zeta(x, N):
    sigma = nn.Parameter(x)
    binomial_coefficient=None
    M = sigma.shape[0]
    sigma_u = sigma.unsqueeze(0).expand(N,M)
    if(binomial_coefficient is None):
        # we compute coeficient
        v = torch.arange(N)
        v[0] = 1
        n_fact = v.prod()
        k_fact = torch.cat([v[:i].prod().unsqueeze(0) for i in range(1, v.shape[0]+1)],0)
        nmk_fact = k_fact.flip(0)
        binomial_coefficient = n_fact/(k_fact * nmk_fact)  
    binomial_coefficient = binomial_coefficient.unsqueeze(-1).expand(N,M).double()

    range_ = torch.arange(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()
    ones_ = torch.ones(N ,device=sigma.device).unsqueeze(-1).expand(N,M).double()

    alternate_neg = (-ones_)**(range_)

    ins = (((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2)
    ins_squared = ((((N-1) - 2 * range_)  * sigma_u)/math.sqrt(2))**2
    as_o = (1+torch.erf(ins)) * torch.exp(ins_squared)
    bs_o = binomial_coefficient * as_o
    r = alternate_neg * bs_o    


    zeta = ZETA_CST * sigma * r.sum(0) * (1/(2**(N-1)))

    return zeta  * surface(N - 1)

class ZetaPhiStorage(object):

    def __init__(self, sigma, N):
        self.N = N
        self.sigma = sigma
        self.m_zeta_var = (zeta(sigma, N)).detach()

        c1 = self.m_zeta_var.sum() != self.m_zeta_var.sum() 
        c2 = self.m_zeta_var.sum() == math.inf
        c3 = self.m_zeta_var.sum() == -math.inf
        if(c1 or c2 or c3):
            print("WARNING : ZetaPhiStorage , untracktable normalisation factor :")
            max_nf = len(sigma)

            limit_nf = ((self.m_zeta_var/self.m_zeta_var) * 0).nonzero()[0].item()
            self.sigma = self.sigma[0:limit_nf]
            self.m_zeta_var = self.m_zeta_var[0:limit_nf]
            if(c1):
                print("\t Nan value in processing normalisation factor")
            if(c2 or c3):
                print("\t +-inf value in processing normalisation factor")
            
            print("\t Max variance is now : ", self.sigma[-1])
            print("\t Number of possible variance is now : "+str(len(self.sigma))+"/"+str(max_nf))

        self.phi_inv_var = (self.sigma**3 * log_grad_zeta(self.sigma, N)).detach()
        
    def zeta(self, sigma):
        N, P = sigma.shape[0], self.sigma.shape[0]
        ref = self.sigma.unsqueeze(0).expand(N, P)
        val = sigma.unsqueeze(1).expand(N,P)
        values, index = torch.abs(ref - val).min(-1)
        
        return self.m_zeta_var[index]

    def inverse_phi(self, sigma):
        return sigma**3 * log_grad_zeta(sigma, self.N).detach()

    def phi(self, phi_val):
        N, P = phi_val.shape[0], self.sigma.shape[0]
        ref = self.phi_inv_var.unsqueeze(0).expand(N, P)
        val = phi_val.unsqueeze(1).expand(N,P)
        # print("val ", val)
        values, index = torch.abs(ref - val).min(-1)
        return self.sigma[index]
    
    def to(self, device):
        self.sigma = self.sigma.to(device)
        self.m_zeta_var = self.m_zeta_var.to(device)
        self.phi_inv_var = self.phi_inv_var.to(device)


class CategoricalDistributionSampler(object):
    def __init__(self, distribution, n=100):
        self.distribution_prob = distribution
        self.sampling_table =  []
        for i, p in enumerate(self.distribution_prob):
            if(p*len(distribution) * n > 1):
                self.sampling_table.append((torch.ones(int(p*len(distribution) * n)) * i).long())
        self.sampling_table = torch.cat(self.sampling_table,0)
    
    def cuda(self):
        self.sampling_table = self.sampling_table.cuda()
    
    def cpu(self):
        self.sampling_table = self.sampling_table.cpu()

    def to(self, device):
        self.sampling_table = self.sampling_table.to(device)

    def sample(self, sample_shape=(1,)):
        rand_index = (torch.rand(sample_shape, device=self.sampling_table.device).view(-1) * len(self.sampling_table)).long()
        return self.sampling_table[rand_index].view(sample_shape)
    

def euclidean_norm_factor(sigma):

    return ((2*math.pi) * sigma)

def gaussianPDF(x, mu, sigma, distance=pf.distance, norm_func=zeta):
    N, D, M = x.shape + (mu.shape[0],)
    x_rd = x.unsqueeze(1).expand(N, M, D)
    mu_rd = mu.unsqueeze(0).expand(N, M, D)
    sigma_rd = sigma.unsqueeze(0).expand(N, M)
    num = torch.exp(-((distance(x_rd, mu_rd)**2))/(2*(sigma_rd)**2))
    den = norm_func(sigma)
    return num/den.unsqueeze(0).expand(N, M)

####################################### TESTING #####################################

def test_zeta_phi_storage():
    torch.set_default_tensor_type(torch.DoubleTensor)
    sigma_s = torch.arange(5e-2, 2., 0.01)
    ZPS = ZetaPhiStorage(sigma_s, 3)
    inverse_phi = ZPS.phi_inv_var
    assert((sigma_s == ZPS.phi(inverse_phi)).float().mean() == 1)

if __name__ == "__main__":
    test_zeta_phi_storage()