import argparse
import tqdm
import random 
import os

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from rcome.manifold.poincare_ball import PoincareBallExact
from rcome.clustering_tools.poincare_em import PoincareEM 
from rcome.data_tools import data_loader, corpora, corpora_tools
from rcome.embedding_tools.losses import graph_embedding_criterion, graph_community_criterion
from rcome.visualisation_tools.plot_tools import plot_poincare_disc_embeddings, plot_poincare_gmm
from rcome.function_tools.distribution_function import CategoricalDistributionSampler
from rcome.optim_tools import rsgd


parser = argparse.ArgumentParser(description='Start an experiment')

parser.add_argument('--initial-lr', dest="initial_lr", type=float, default=1e-1,
                    help="Learning rate for the first embedding step")

# embedding parameters
parser.add_argument('--dim', dest="dim", type=int, default=2,
                    help="dimensionality of emebdding")

# Main learning loop parameters
parser.add_argument('--lr', dest="lr", type=float, default=1e-2,
                    help="learning rate for embedding")
parser.add_argument('--alpha', dest="alpha", type=float, default=1,
                    help="alpha for embedding")
parser.add_argument('--beta', dest="beta", type=float, default=1,
                    help="beta for embedding")
parser.add_argument('--gamma', dest="gamma", type=float, default=1e-1,
                    help="gamma rate for embedding")
parser.add_argument('--n-negative', dest='n_negative', type=int, default=10,
                    help='number of negative examples in L_2 loss')
parser.add_argument('--epoch', dest="epoch", type=int, default=10,
                    help="number of loops alternating embedding/EM")
parser.add_argument("--batch-size", dest="batch_size", type=int, default=20,
                    help="batch number of elements")

# Data parameters 
parser.add_argument('--dataset', dest="dataset", type=str, default="LFR1",
                    help="dataset to use for the experiments")
parser.add_argument('--walk-by-node', dest="walk_by_node", type=int, default=2,
                    help="size of random walk")
parser.add_argument('--walk-lenght', dest="walk_lenght", type=int, default=10,
                    help="size of random walk")
parser.add_argument('--context-size', dest="context_size", type=int, default=3,
                    help="size of the context used on the random walk")


# others
parser.add_argument('--verbose', dest="verbose", default=False, action="store_true",
                    help="Do print information")   
parser.add_argument('--plot', dest="plot", default=False, action="store_true",
                    help="draw figures at each iteration (can be time consuming)")
                    
# Optimisation
parser.add_argument('--cuda', dest="cuda", action="store_true", default=False,
                    help="Optimize on GPU (nvidia only)")
parser.add_argument('--num-threads', dest="num_threads", type=int, default=1,
                    help="Number of threads for pytorch dataloader (can fail on windows)")
args = parser.parse_args()

torch.set_default_tensor_type(torch.DoubleTensor)

if(args.verbose):
    print("Loading Corpus ")
X, Y = data_loader.load_corpus(args.dataset, directed=False)

dataset_l1 = corpora.NeigbhorFlatCorpus(X, Y)
dataset_l2 = corpora.RandomContextSizeFlat(X, Y, precompute=args.walk_by_node, 
                path_len=args.walk_lenght, context_size=args.context_size)
dataset_l3 = corpora_tools.from_indexable(torch.arange(0, len(X), 1).unsqueeze(-1))

dataloader_l1 = DataLoader(dataset_l1, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            drop_last=False,
                            num_workers= args.num_threads if(args.num_threads > 1) else 0
                    )

def collate_fn_simple(tensor_list):
    v =  torch.cat(tensor_list, 0)
    return v[:,0], v[:,1]

dataloader_l2 = DataLoader(dataset_l2, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_simple,
                            num_workers= args.num_threads if(args.num_threads > 1) else 0)

dataloader_l3 = DataLoader(dataset_l3, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            drop_last=False,
                            num_workers= args.num_threads if(args.num_threads > 1) else 0
                    )

frequency = dataset_l2.getFrequency()
idx = frequency[:,0].sort()[1]
frequency = frequency[idx]**(3/4)
frequency[:,1] /= frequency[:,1].sum()
distribution = CategoricalDistributionSampler(frequency[:,1])
n_community = max([max(communities) for key, communities in Y.items()]) + 1

if(args.verbose):
    print('Number of communities', n_community)
    print("Initialise embedding")
node_embedding = nn.Embedding(len(X), args.dim, max_norm=0.999)
node_embedding.weight.data[:] = node_embedding.weight.data * 1e-2
context_embedding = nn.Embedding(len(X), args.dim, max_norm=0.999)
context_embedding.weight.data[:] = context_embedding.weight.data * 1e-2

if(args.cuda):
    node_embedding.cuda()
    context_embedding.cuda()
    memory_transfer = lambda x: x.cuda()

else:
    memory_transfer = lambda x: x

if(args.verbose):
    print("Optimisation and manifold intialisation")
manifold = PoincareBallExact
optimizer_init = rsgd.RSGD(list(node_embedding.parameters()) + list(context_embedding.parameters()), args.initial_lr, manifold=manifold)


# Initialise embedding 
if(args.verbose):
    print("Initialise embedding")

for i in range(20):

    l2 = 0.
    for x, y in dataloader_l2:
        optimizer_init.zero_grad()
        pe_x = node_embedding(memory_transfer(x.long()))
        pe_y = context_embedding(memory_transfer(y.long()))
        ne = context_embedding(memory_transfer(distribution.sample(sample_shape=(len(x), args.n_negative)))).detach()
        loss = args.beta * graph_embedding_criterion(pe_x, pe_y, z=ne, manifold=manifold).sum()
        l2 += loss.item()
        loss.backward()
        optimizer_init.step()

    l1 = 0.
    for x, y in dataloader_l1:

        optimizer_init.zero_grad()
        pe_x = memory_transfer(node_embedding(x.long()))
        pe_y = memory_transfer(node_embedding(y.long()))
        loss = args.alpha * graph_embedding_criterion(pe_x, pe_y, manifold=manifold).sum()
        l1 += loss.item()
        loss.backward()
        optimizer_init.step()

    print('Loss value for iteration ', i,' is ', l1/len(dataset_l1) + l2/len(dataset_l2))
if(args.plot):
    plot_poincare_disc_embeddings(node_embedding.weight.data,  labels=Y, save_folder="LOG/all_loss", file_name="rcome_embedding_script_gmm_init.png")

optimizer = rsgd.RSGD(list(node_embedding.parameters()) + list(context_embedding.parameters()), args.lr, manifold=manifold)
# main learning loop
for i in range(args.epoch):

    l1 = 0.
    for x, y in dataloader_l1:

        optimizer.zero_grad()
        pe_x = node_embedding(memory_transfer(x.long()))
        pe_y = node_embedding(memory_transfer(y.long()))

        loss = args.alpha * graph_embedding_criterion(pe_x, pe_y, manifold=manifold).sum()
        l1 += loss.item()
        loss.backward()
        optimizer.step()

    l2 = 0.
    for x, y in dataloader_l2:
        optimizer.zero_grad()
        pe_x = memory_transfer(node_embedding(x.long()))
        pe_y = memory_transfer(context_embedding(y.long()))
        ne = context_embedding(memory_transfer(distribution.sample(sample_shape=(len(x), args.n_negative)))).detach()
        loss = args.beta * graph_embedding_criterion(pe_x, pe_y, z=ne, manifold=manifold).sum()
        l2 += loss.item()
        loss.backward()
        optimizer.step()


    em_alg = PoincareEM(n_community)
    em_alg.fit(memory_transfer(node_embedding.weight.data))

    NF = em_alg.get_normalisation_coef()
    pi, mu, sigma = em_alg.get_parameters()

    pik = em_alg.get_pik(node_embedding.weight.data)

    l3 = 0.

    for x in dataloader_l3:
        optimizer.zero_grad()

        pe_x = node_embedding(memory_transfer(x[0].long()))
        wik = pik[memory_transfer(x[0].long())]
        loss = args.gamma* graph_community_criterion(pe_x.squeeze(), wik.detach(), mu.detach(), sigma.detach(), NF.detach(), manifold=manifold).sum()
        l3 += loss.item()
        loss.backward()
        optimizer.step()

    print('L_1 + L_2 value for iteration ', i,' is ', l1/len(dataset_l1) + l2/len(dataset_l2) )
    print('L_3 value', l3/len(dataset_l3))
    if(args.plot):
        plot_poincare_gmm(node_embedding.weight.data, em_alg , labels=Y, save_folder="LOG/all_loss", file_name="rcome_embedding_script_gmm"+str(i)+".png")
