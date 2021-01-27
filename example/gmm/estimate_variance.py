import torch
from rcome.function_tools.distribution_function import ZetaPhiStorage, zeta_disc

dim = 2
sigma_values = torch.arange(1e-2, 20, 0.0001)
print(sigma_values.shape)
ZPS = ZetaPhiStorage(sigma_values, dim)

squared_distance = torch.Tensor([1, 2, 10, 20, 50, 2000])

sigma_estimation = ZPS.phi(squared_distance)

print("Estimation of sigma ", sigma_estimation.tolist())
print("Normalisation coeficient ", ZPS.zeta(sigma_estimation).detach())
print("Normalisation coeficient 2D ", zeta_disc(sigma_estimation))
print("Inverse phi given ", squared_distance.tolist())
print("Inverse phi estimated ", ZPS.inverse_phi(sigma_estimation).tolist())
