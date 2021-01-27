'''
Poincare GMM estimation through Expectation-Maximisation algorithm
for the Poincare Ball model
'''

import math
import cmath
import torch
import numpy as np
import tqdm


from rcome.clustering_tools import poincare_kmeans as kmh
from rcome.function_tools import distribution_function as df
from rcome.function_tools import poincare_function as pf
from rcome.function_tools import poincare_alg as pa


class PoincareEM(object):
    def __init__(self, n_gaussian, init_mod="kmeans-hyperbolic", verbose=False):
        self._n_g = n_gaussian
        self._distance = pf.distance

        self._verbose = verbose
        self._init_mod = init_mod
        self._started = False
        self.cwik = None

    def update_w(self, z, wik, g_index=-1):
        with torch.no_grad():
            if(g_index > 0):
                self._w[g_index] = wik[:, g_index].mean()
            else:
                self._w = wik.mean(0)

    def update_mu(self, z, wik, lr_mu, tau_mu, g_index=-1, max_iter=150):
        N, D, M = z.shape + (wik.shape[-1],)
        # if too much gaussian we compute mean for each gaussian separately (To avoid too large memory)
        if(M>40):
            for g_index in range(M//40 + (1 if(M%40 != 0) else 0)):
                from_ = g_index * 40
                to_ = min((g_index+1) * 40, M)


                zz = z.unsqueeze(1).expand(N, to_-from_, D)
                self._mu[from_:to_] = pa.barycenter(zz, wik[:, from_:to_], lr_mu, tau_mu, max_iter=max_iter, 
                                                    verbose=True, normed=True).squeeze()
        else:   
            if(g_index>=0):
                self._mu[g_index] = pa.barycenter(z, wik[:, g_index], lr_mu, tau_mu, max_iter=max_iter, normed=True).squeeze()
            else:
                self._mu = pa.barycenter(z.unsqueeze(1).expand(N, M, D), wik, lr_mu,  tau_mu, max_iter=max_iter, normed=True).squeeze()

    def update_sigma(self, z, wik, g_index=-1):
        with torch.no_grad():
            N, D, M = z.shape + (self._mu.shape[0],)
            if(g_index>0):
                dtm = ((self._distance(z, self._mu[:,g_index].expand(N))**2) * wik[:, g_index]).sum()/wik[:, g_index].sum()
                self._sigma[:, g_index] = self.phi(dtm)
            else:
                dtm = ((self._distance(z.unsqueeze(1).expand(N,M,D), self._mu.unsqueeze(0).expand(N,M,D))**2) * wik).sum(0)/wik.sum(0)

                self._sigma = self.zeta_phi.phi(dtm)        

    def _expectation(self, z):
        with torch.no_grad():
            # computing wik 
            pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
            if(pdf.mean() != pdf.mean()):
                print("EXPECTATION : pdf contain not a number elements")
                quit()
            p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)

            # it can happens sometime to get (due to machine precision) a node with all pdf to zero
            # in this case we must detect it. There is severall solution, in our case we affect it
            # equally to each gaussian. 
            if(p_pdf.sum(-1).min() <= 1e-15):
                if(self._verbose):    
                    print("EXPECTATION : pdf.sum(-1) contain zero for ", (p_pdf.sum(-1)<= 1e-15).sum().item(), "items")
                p_pdf[p_pdf.sum(-1) <= 1e-15] = 1
                
            wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
            if(wik.mean() != wik.mean()):
                print("EXPECTATION : wik contain not a number elements")
                quit()
            # print(wik.mean(0))
            if(wik.sum(1).mean() <= 1-1e-4 and wik.sum(1).mean() >= 1+1e-4 ):
                print("EXPECTATION : wik don't sum to 1")
                print(wik.sum(1))
                quit()
            return wik

    def _maximization(self, z, wik, lr_mu=1e-4, tau_mu=1e-5, max_iter_bar=math.inf):
        self.update_w(z, wik)
        if(self._w.mean() != self._w.mean()):
            print("UPDATE : w contain not a number elements")
            quit()            
        # print("w", self._w)
        self.update_mu(z, wik, lr_mu=lr_mu, tau_mu=tau_mu, max_iter=max_iter_bar)
        if(self._verbose):
            print(self._mu)
        if(self._mu.mean() != self._mu.mean()):
            print("UPDATE : mu contain not a number elements")
            quit()
        if(self._verbose):
            print("sigma b", self._sigma) 
        self.update_sigma(z, wik)
        if(self._verbose):
            print("sigma ", self._sigma)
        if(self._sigma.mean() != self._sigma.mean()):
            print("UPDATE : sigma contain not a number elements")
            quit()  


    def fit(self, z, max_iter=200, lr_mu=5e-2, tau_mu=1e-4, Y=None):
        # if first time fit is called init all parameters

        if(not self._started):
            self._d = z.size(-1)
            self._mu = (torch.rand(self._n_g,self._d ) - 0.5)/self._d 
            self._sigma = torch.rand(self._n_g)/10 +0.8
            self._w = torch.ones(self._n_g)/self._n_g
            # print(self._w.dtype)
            if(self._verbose):
                print("size",z.size() )
            self.zeta_phi = df.ZetaPhiStorage(torch.arange(1e-2, 2., 0.001), self._d)
        
        if(Y is not None):

            self._mu = self._mu.to(z.device)
            self._sigma = self._sigma.to(z.device)
            self._w = self._w.to(z.device)
            self.zeta_phi.to(z.device)
            wik = Y.float().to(z.device)
            self.real_Y = Y.sum(0).nonzero().squeeze()
            wik = wik[:, self.real_Y]
            old_ng = self._n_g
            self._n_g = self.real_Y.shape[0]
            self._mu = self._mu[self.real_Y]
            self._w = self._w[self.real_Y]
            self._sigma = self._sigma[self.real_Y]

            self._maximization(z, wik, lr_mu=lr_mu, tau_mu=tau_mu)

            self._n_g = old_ng
            self._mu_reshaped = torch.zeros(self._n_g, self._mu.size(-1)).to(z.device)
            self._mu_reshaped[self.real_Y] = self._mu
            self._mu = self._mu_reshaped

            self._w_reshaped = torch.zeros(self._n_g).to(z.device)
            self._w_reshaped[self.real_Y] = self._w
            self._w = self._w_reshaped

            self._sigma_reshaped = torch.zeros(self._n_g).to(z.device)
            self._sigma_reshaped[self.real_Y] = self._sigma
            self._sigma = self._sigma_reshaped
            return
        else:
            progress_bar = tqdm.trange(max_iter) if(self._verbose) else range(max_iter)
            # if it is the first time function fit is called
            if(not self._started):
                # using kmeans for initializing means
                if(self._init_mod == "kmeans-hyperbolic"):
                    if(self._verbose):
                        print("Initialize means using kmeans hyperbolic algorithm")
                    km = kmh.PoincareKMeansNInit(self._n_g, n_init=2)
                    km.fit(z)
                    self._mu = km.cluster_centers_
            if(self._verbose):
                print("\t mu -> ", self._mu)
                print("\t sigma -> ", self._sigma)
            self._started = True
        # set in the z device
        self._mu = self._mu.to(z.device)
        self._sigma = self._sigma.to(z.device)
        self._w = self._w.to(z.device)
        self.zeta_phi.to(z.device)
        wik = torch.ones(z.size(0), self._mu.size(0)).to(self._mu.device)
        for epoch in progress_bar:
            old_wik = wik
            wik = self._expectation(z)
            if((old_wik-wik).abs().mean() < 1e-4 and epoch>10):
                return
            
            self._maximization(z, wik, lr_mu=lr_mu, tau_mu=1e-5)
        print("WARNING :  EM did not converge, consider increasing the number of iteration")
        self.cwik = wik

    def get_parameters(self):
        return  self._w, self._mu, self._sigma
        
    def set_parameters(self, pi, mu, sigma):
        self._d = mu.size(-1)
        self.zeta_phi = df.ZetaPhiStorage(torch.arange(5e-3, 2., 0.001), self._d)
        self.zeta_phi.to(pi.device)
        self._w = pi
        self._mu = mu
        self._sigma = sigma

    def get_pik(self, z):
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        if(p_pdf.sum(-1).min() == 0):
            print("EXPECTATION (function get_pik) : pdf.sum(-1) contain zero for ", (p_pdf.sum(-1)<= 1e-15).sum().item(), "items" )
            p_pdf[p_pdf.sum(-1) == 0] = 1e-8
        wik = p_pdf/p_pdf.sum(-1, keepdim=True).expand_as(pdf)
        return wik  
    
    def get_normalisation_coef(self):
        return self.zeta_phi.zeta(self._sigma)

    def predict(self, z):
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        return p_pdf.max(-1)[1]

    def get_unormalized_probs(self, z):
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        p_pdf = pdf * self._w.unsqueeze(0).expand_as(pdf)
        return p_pdf

    def get_density(self, z):
        N, D, M = z.shape + (self._mu.shape[0],)
        pdf = df.gaussianPDF(z, self._mu, self._sigma, norm_func=self.zeta_phi.zeta) 
        density = (pdf * self._w.unsqueeze(0).expand_as(pdf)).sum(-1)
        return density

    
    def probs(self, z):
        return self.get_pik(z)
