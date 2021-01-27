import torch
from rcome.manifold import poincare_ball

# a point in the ball
point = torch.rand(1,2) - 0.5
# a vector
vector =  torch.rand(1,2)
pb_manifold = poincare_ball.PoincareBallExact()

projected_point = pb_manifold.riemannian_exp(point, vector)
vector_2 = pb_manifold.riemannian_log(point, projected_point)

print('Random point', point, '\n'
      'Random vector', vector, '\n'
      'Projected point obtained via Exponential map of ', '\n'
      'random point in the direction of random vector', projected_point, '\n'
      'Logarithmic map from the Projected point to the ', '\n'
      'random point gives the random vector', vector_2)