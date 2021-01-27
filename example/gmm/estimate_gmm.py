# This script estimate gmm 
import torch
from rcome.clustering_tools import poincare_em
from rcome.visualisation_tools.plot_tools import plot_poincare_gmm
# generating random set draw fro uniform
set_A = torch.randn(50,2)*2e-1 + 0.3

set_B = torch.randn(50,2)*1e-1 - 0.5

representation = torch.cat((set_A, set_B), 0)
out_of_disc =  (representation[representation.norm(2,-1)>1].norm(2,-1) + 1e-2)
representation[representation.norm(2,-1)>1] = \
    torch.einsum('ij, i -> ij', representation[representation.norm(2,-1)>1], 1/out_of_disc)


# estimate the gmm
em_alg = poincare_em.PoincareEM(2)
em_alg.fit(representation)

Y = torch.zeros(100,1).long()
Y[:50] = 1

plot_poincare_gmm(representation, em_alg , labels=Y, save_folder="LOG/gmm_estimation", file_name="gmm_estimation.png")
