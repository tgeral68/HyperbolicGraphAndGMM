import torch
import os
from torch import nn
from torch.utils.data import DataLoader

from rcome.manifold.poincare_ball import PoincareBallApproximation, PoincareBallExact
from rcome.optim_tools import rsgd
from rcome.data_tools import data_loader, corpora
from rcome.function_tools.distribution_function import CategoricalDistributionSampler
from rcome.embedding_tools.losses import graph_embedding_criterion

from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings

X, Y = data_loader.load_corpus("LFR1", directed=False)
dataset = corpora.RandomContextSizeFlat(X, Y, precompute=2, 
                path_len=10, context_size=3)

def collate_fn_simple(my_list):
    v =  torch.cat(my_list,0)
    return v[:,0], v[:,1]

dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn_simple)

model = nn.Embedding(len(X), 2, max_norm=0.999)
model.weight.data[:] = model.weight.data * 1e-2
model_context = nn.Embedding(len(X), 2, max_norm=0.999)
model_context.weight.data[:] = model.weight.data * 1e-2

manifold = PoincareBallExact
optimizer = rsgd.RSGD(list(model.parameters()) + list(model_context.parameters()), 1e-1, manifold=manifold)
default_gt = torch.zeros(20).long()

# negative sampling distribution
frequency = dataset.getFrequency()
idx = frequency[:,0].sort()[1]
frequency = frequency[idx]**(3/4)
frequency[:,1] /= frequency[:,1].sum()

distribution =  CategoricalDistributionSampler(frequency[:,1])


for i in range(50):
    tloss = 0.
    for x, y in dataloader:
        optimizer.zero_grad()
        pe_x = model(x.long())
        pe_y = model_context(y.long())
        ne = model_context(distribution.sample(sample_shape=(len(x), 10))).detach()
        loss = graph_embedding_criterion(pe_x, pe_y, z=ne, manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()
    print('Loss value for iteration ', i,' is ', tloss)


plot_poincare_disc_embeddings(model.weight.data.numpy(),
                              labels=dataset.Y,
                              save_folder="LOG/second_order",
                              file_name="LFR_second_order_negative.png")

torch.save(model.weight.data, "LOG/second_order/representation.pth")
