import torch
import os
from torch import nn
from torch.utils.data import DataLoader

from rcome.manifold.poincare_ball import PoincareBallApproximation, PoincareBallExact
from rcome.optim_tools import rsgd
from rcome.data_tools import data_loader, corpora, corpora_tools
from rcome.embedding_tools.losses import graph_embedding_criterion
from rcome.embedding_tools.losses import graph_community_criterion
from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings

from rcome.clustering_tools import poincare_em

X, Y = data_loader.load_corpus("LFR1", directed=False)

dataset = corpora.NeigbhorFlatCorpus(X, Y)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = nn.Embedding(len(X), 2, max_norm=0.999)
model.weight.data[:] = model.weight.data * 1e-2

manifold = PoincareBallExact
optimizer = rsgd.RSGD(model.parameters(), 1e-1, manifold=manifold)
default_gt = torch.zeros(20).long()
criterion = nn.CrossEntropyLoss(reduction="sum")


for i in range(10):
    tloss = 0.
    for x, y in dataloader:

        optimizer.zero_grad()
        pe_x = model(x.long())
        pe_y = model(y.long())
        ne = model((torch.rand(len(x), 10) * len(X)).long())
        loss = graph_embedding_criterion(pe_x, pe_y, z=ne, manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()
    print('Loss value for iteration ', i,' is ', tloss)


em_alg = poincare_em.PoincareEM(13)
em_alg.fit(model.weight.data)

NF = em_alg.get_normalisation_coef()
pi, mu, sigma = em_alg.get_parameters()
pik = em_alg.get_pik(model.weight.data)

plot_poincare_disc_embeddings(model.weight.data.numpy(), labels=dataset.Y, save_folder="LOG/community_loss", file_name="LFR_before_community_loss.png", centroids=mu)

optimizer = rsgd.RSGD(model.parameters(), 2e-2, manifold=manifold)

dataset_index = corpora_tools.from_indexable(torch.arange(0,len(X),1).unsqueeze(-1))
dataloader = DataLoader(dataset_index, 
                            batch_size=10, 
                            shuffle=True,
                            drop_last=False
                    )

for i in range(20):
    tloss = 0.
    pik = em_alg.get_pik(model.weight.data)
    for x in dataloader:
        optimizer.zero_grad()

        pe_x = model(x[0].long())
        wik = pik[x[0].long()]
        loss = graph_community_criterion(pe_x.squeeze(), wik.detach(), mu.detach(), sigma.detach(), NF.detach(), manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()
    plot_poincare_disc_embeddings(model.weight.data.numpy(), labels=dataset.Y, save_folder="LOG/community_loss", file_name="LFR_after_community_loss"+str(i)+".png", centroids=mu)
    print('Loss value for iteration ', i,' is ', tloss)

plot_poincare_disc_embeddings(model.weight.data.numpy(), labels=dataset.Y, save_folder="LOG/community_loss", file_name="LFR_after_community_loss.png", centroids=mu)