''' Example plotting a Gaussian distribution on hyperbolic space.
'''
from matplotlib import pyplot as plt
import torch

from rcome.function_tools import distribution_function as df

# Define the barycentre mu and variance sigma
mu = torch.Tensor([[0.7, 0.7]])
sigma = torch.Tensor([2.2])

# Precompute the normalisation factor
norm_factor = df.ZetaPhiStorage(torch.arange(5e-2, 2., 0.001), 2)

# Sample a number of points on the manifold
points = (torch.rand(200000, 2) - 0.5) * 2
points = points[points.norm(2, -1) < 0.9999]

#Compute the probability density of each point
probs = df.gaussianPDF(points, mu, sigma, norm_func=norm_factor.zeta).squeeze()

#Divide into two classes
points_high = points[probs > 0.001]
points_med = points[probs > 0.003]
points_low = points[probs <= 0.003]

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.axis('off')
plt.scatter(points_low[:, 0].numpy(), points_low[:, 1].numpy(), c='red')
plt.scatter(points_high[:, 0].numpy(), points_high[:, 1].numpy(), c='blue')
plt.scatter(points_med[:, 0].numpy(), points_med[:, 1].numpy(), c='green')
plt.scatter(mu[:, 0].numpy(),mu[:, 1].numpy(), c='orange', marker='*', s=150)
plt.savefig('gaussian.png')
plt.show()