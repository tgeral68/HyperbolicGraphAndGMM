import torch
import os

from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings
from rcome.function_tools import poincare_alg as pa
from matplotlib import pyplot as plt

data_point = (torch.randn(50,2)/5 +0.6)
print(data_point)
norms = data_point.norm(2,-1) 
data_point[norms >= 1.] = torch.einsum('ij, i -> ij', data_point[norms >= 1.], 1/(norms[norms >= 1.] + 1e-5))
data_point.norm(2,-1)

barycenter_hyperbolic = pa.barycenter(data_point, verbose=True)

print(barycenter_hyperbolic)
plot_poincare_disc_embeddings(data_point.numpy(), close=False,
                              save_folder="LOG/mean", file_name="example_mean_hyperbolic.png")

barycenter_euclidean = data_point.mean(0, keepdim=True)

plt.scatter(barycenter_hyperbolic[:, 0], barycenter_hyperbolic[:,1],marker='D', s=300., c='red', label="Hyperbolic")  
plt.scatter(barycenter_euclidean[:, 0], barycenter_euclidean[:,1],marker='D', s=300., c='green', label="Euclidean") 
plt.legend(prop={'size': 25})
plt.savefig("LOG/mean/example_mean_hyperbolic_euclidean.png", format="png")