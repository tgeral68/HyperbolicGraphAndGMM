import torch
import os
from torch import nn
from torch.utils.data import DataLoader

from rcome.manifold.poincare_ball import PoincareBallApproximation, PoincareBallExact
from rcome.optim_tools import rsgd
from rcome.data_tools import data_loader, corpora
from rcome.embedding_tools.losses import tree_embedding_criterion

from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings

X, Y = data_loader.load_corpus("dblp", directed=False)

dataset = corpora.NeigbhorFlatCorpus(X, Y)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

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


plot_poincare_disc_embeddings(model.weight.data.cpu().numpy(), labels=dataset.Y, save_folder="LOG/first_order", file_name="DBLP_hierarchical.png")
