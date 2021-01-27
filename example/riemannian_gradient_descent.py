import torch
from torch import nn
from torch.utils.data import DataLoader

from rcome.optim_tools.rsgd import RSGD

# loading dataset
X, Y = data_loader.load_corpus(dataset_name, directed)
# define organisation of data
...

# otpimisation instanciation
manifold = PoincareBallExact
optimizer = RSGD(model_parameters, learning_rate, manifold)

# learning loop
for i in range(total_iteration):
    for x, y in dataloader:

        optimizer.zero_grad()
        # define the loss
        ...
        loss.backward()
        optimizer.step()