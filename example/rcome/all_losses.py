import torch
import os
from torch import nn
from torch.utils.data import DataLoader

from rcome.manifold.poincare_ball import PoincareBallApproximation, PoincareBallExact
from rcome.optim_tools import rsgd
from rcome.data_tools import data_loader, corpora, corpora_tools
from rcome.embedding_tools.losses import graph_embedding_criterion
from rcome.embedding_tools.losses import graph_community_criterion
from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings, plot_poincare_gmm
from rcome.function_tools.distribution_function import CategoricalDistributionSampler


from rcome.clustering_tools import poincare_em
from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings

X, Y = data_loader.load_corpus("LFR1", directed=False)

dataset = corpora.NeigbhorFlatCorpus(X, Y)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = nn.Embedding(len(X), 2, max_norm=0.999)
model.weight.data[:] = model.weight.data * 1e-2

manifold = PoincareBallExact
optimizer = rsgd.RSGD(model.parameters(), 1e-1, manifold=manifold)
default_gt = torch.zeros(20).long()
criterion = nn.CrossEntropyLoss(reduction="sum")


for i in range(50):
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

dataset_o3 = corpora_tools.from_indexable(torch.arange(0,len(X),1).unsqueeze(-1))
dataloader_o3 = DataLoader(dataset_o3, 
                            batch_size=10, 
                            shuffle=True,
                            drop_last=False
                    )

dataset_o1 = corpora.NeigbhorFlatCorpus(X, Y)
dataloader_o1 = DataLoader(dataset_o1, 
                            batch_size=10, 
                            shuffle=True,
                            drop_last=False
                    )

dataset_o2 = corpora.RandomContextSizeFlat(X, Y, precompute=2, 
                path_len=10, context_size=3)


def collate_fn_simple(my_list):
    v =  torch.cat(my_list,0)
    return v[:,0], v[:,1]

dataloader_o2 = DataLoader(dataset_o2, batch_size=5, shuffle=True, collate_fn=collate_fn_simple)

model_context = nn.Embedding(len(X), 2, max_norm=0.999)
model_context.weight.data[:] = model.weight.data * 1e-2

optimizer = rsgd.RSGD(list(model.parameters()) + list(model_context.parameters()), 5e-3, manifold=manifold)

default_gt = torch.zeros(20).long()

# negative sampling distribution
frequency = dataset_o2.getFrequency()
idx = frequency[:,0].sort()[1]
frequency = frequency[idx]**(3/4)
frequency[:,1] /= frequency[:,1].sum()

distribution =  CategoricalDistributionSampler(frequency[:,1])

plot_poincare_disc_embeddings(model.weight.data.numpy(), labels=dataset.Y, save_folder="LOG/all_loss", file_name="LFR_all_loss_init.png", centroids=mu)

for i in range(5):
    tloss = 0.




    for x, y in dataloader_o2:
        optimizer.zero_grad()
        pe_x = model(x.long())
        pe_y = model_context(y.long())
        ne = model_context(distribution.sample(sample_shape=(len(x), 10))).detach()
        loss = graph_embedding_criterion(pe_x, pe_y, z=ne, manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()

    for x, y in dataloader_o1:

        optimizer.zero_grad()
        pe_x = model(x.long())
        pe_y = model(y.long())
        ne = model((torch.rand(len(x), 10) * len(X)).long())

        loss = graph_embedding_criterion(pe_x, pe_y, manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()

    em_alg = poincare_em.PoincareEM(13)
    em_alg.fit(model.weight.data)

    NF = em_alg.get_normalisation_coef()
    pi, mu, sigma = em_alg.get_parameters()

    pik = em_alg.get_pik(model.weight.data)

    for x in dataloader_o3:
        optimizer.zero_grad()

        pe_x = model(x[0].long())
        wik = pik[x[0].long()]
        loss = 1e-1 * graph_community_criterion(pe_x.squeeze(), wik.detach(), mu.detach(), sigma.detach(), NF.detach(), manifold=manifold).sum()
        tloss += loss.item()
        loss.backward()
        optimizer.step()

    plot_poincare_gmm(model.weight.data, em_alg , labels=dataset.Y, save_folder="LOG/all_loss", file_name="LFR_all_loss_gmm"+str(i)+".png")
    plot_poincare_disc_embeddings(model.weight.data.numpy(), labels=dataset.Y, save_folder="LOG/all_loss", file_name="LFR_all_loss"+str(i)+".png", centroids=mu)
    print('Loss value for iteration ', i,' is ', tloss)

plot_poincare_disc_embeddings(model.weight.data.numpy(), labels=dataset.Y, save_folder="LOG/all_loss", file_name="LFR_all_loss.png", centroids=mu)

plot_poincare_gmm(model.weight.data, em_alg , labels=dataset.Y, save_folder="LOG/all_loss", file_name="LFR_all_loss_gmm.png")
torch.save(model.weight.data, "LOG/all_loss/representation.pth")