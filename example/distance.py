import torch

from rcome.manifold import poincare_ball

# points in the ball
point_a = torch.rand(1,2) - 0.5
point_b = torch.rand(1,2) - 0.5
pb_manifold = poincare_ball.PoincareBallExact
distance_a_b = pb_manifold.distance(point_a, point_b)

print('Distance between point ',
      point_a,' and point ', point_b,
      ' is ', distance_a_b)