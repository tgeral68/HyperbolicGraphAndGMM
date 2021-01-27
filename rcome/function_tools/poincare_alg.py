import math
import torch
from torch import nn
from rcome.function_tools import poincare_function as pf

# z and wik must have same dimenssion except if wik is not given
def barycenter(z, wik=None, lr=1e-3, tau=5e-6, max_iter=100, distance=pf.distance, normed=False,
               init_method="default", verbose=False):
    with torch.no_grad():
        if(wik is None):
            wik = 1.
            barycenter = z.mean(0, keepdim=True)
        else:
            wik = wik.unsqueeze(-1).expand_as(z)
            if(init_method == "global_mean"):
                print("Bad init selected")
                barycenter = z.mean(0, keepdim=True)            
            else:
                barycenter = (z*wik).sum(0, keepdim=True)/wik.sum(0)

        if(len(z) == 1):
            return z
        iteration = 0
        cvg = math.inf
        while(cvg>tau and max_iter>iteration):

            iteration+=1
            if(type(wik) != float):
                grad_tangent = 2 * pf.log(barycenter.expand_as(z), z) * wik
                nan_values = (~(barycenter == barycenter))              
                if(torch.nonzero(nan_values.squeeze()).shape[0]>0):
                    print("\n\n A At least one barycenter is Nan : ")
                    print(pf.log(barycenter.expand_as(z), z).sum(0))
                    print("index of nan values ", nan_values.squeeze().nonzero())
                    quit()
                    # torch 1.3 minimum for this operation
                    print("index of nan values ", nan_values.squeeze().nonzero())

            else:
                grad_tangent = 2 * pf.log(barycenter.expand_as(z), z)
            
            if(normed):
                if(type(wik) != float):

                    grad_tangent /= wik.sum(0, keepdim=True).expand_as(wik)
                else:
                    grad_tangent /= len(z)

            cc_barycenter = pf.exp(barycenter, lr * grad_tangent.sum(0, keepdim=True))
            nan_values = (~(cc_barycenter == cc_barycenter))

            if(torch.nonzero(nan_values.squeeze()).shape[0]>0):
                    print("\n\n  At least one barycenter is Nan exp update may contain 0: ")
                    print(grad_tangent.sum(0, keepdim=True))
                    quit()
                    # torch 1.3 minimum for this operation
            cvg = distance(cc_barycenter, barycenter).max().item()

            barycenter = cc_barycenter
            if(cvg<=tau and verbose):
                print("Frechet Mean converged in ", iteration, " iterations")
        return barycenter
