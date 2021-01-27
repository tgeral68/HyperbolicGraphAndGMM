import torch
import os
from torch import nn
from torch.utils.data import DataLoader

from rcome.manifold.poincare_ball import PoincareBallApproximation, PoincareBallExact
from rcome.optim_tools import rsgd
from rcome.data_tools import data_loader, corpora
from rcome.embedding_tools.losses import tree_embedding_criterion

from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings

X, Y = data_loader.load_corpus("LFR1", directed=False)

dataset = corpora.NeigbhorFlatCorpus(X, Y)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = nn.Embedding(len(X), 2, max_norm=0.999)
model.weight.data[:] = model.weight.data * 1e-2
model.cuda()
manifold = PoincareBallExact
optimizer = rsgd.RSGD(model.parameters(), 1e-1, manifold=manifold)
default_gt = torch.zeros(20).long()
criterion = nn.CrossEntropyLoss(reduction="sum")


for i in range(50):
    tloss = 0.
    for x, y in dataloader:

        optimizer.zero_grad()
        pe_x = model(x.long().cuda())
        pe_y = model(y.long().cuda())
        ne = model((torch.rand(len(x), 10) * len(X)).long().cuda())
        loss = tree_embedding_criterion(pe_x, pe_y, z=ne, manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()
    print('Loss value for iteration ', i,' is ', tloss)

from rcome.function_tools import poincare_alg as pa
weigths = torch.Tensor([[ 1 if(y in dataset.Y[i])
                         else 0 for y in range(13)] for i in range(len(X))]).cuda()

barycenters = []
for i in range(13):
    barycenters.append(pa.barycenter(model.weight.data, weigths[:,i], verbose=True).cpu())


plot_poincare_disc_embeddings(model.weight.data.cpu().numpy(), 
                              labels=dataset.Y, centroids=torch.cat(barycenters),
                              save_folder="LOG/mean", file_name="LFR_hierachical.png")
print(barycenters)
barycenters = []
for i in range(13):
    barycenters.append(model.weight.data[weigths[:,i] == 1].mean(0).cpu().unsqueeze(0))

print(barycenters)
plot_poincare_disc_embeddings(model.weight.data.cpu().numpy(), 
                              labels=dataset.Y, centroids=torch.cat(barycenters),
                              save_folder="LOG/mean", file_name="LFR_hierarchical_euclidean.png")