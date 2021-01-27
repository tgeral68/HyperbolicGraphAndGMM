import torch
import os
from torch import nn
from torch.utils.data import DataLoader

from rcome.manifold.poincare_ball import PoincareBallApproximation, PoincareBallExact
from rcome.optim_tools import rsgd
from rcome.data_tools import collections
from rcome.data_tools.corpora import DatasetTuple
from rcome.visualisation_tools.plot_tools import plot_geodesic

root_synset = 'mammal.n.01'
X, dictionary, values, tuple_neigbhor = collections.animals(root=root_synset)
print('Number of nodes', len(X))

dataset = DatasetTuple(X)
dataloader = DataLoader(dataset, batch_size=5)

model = nn.Embedding(len(dictionary)+1, 2, max_norm=0.999)
model.weight.data[:] = model.weight.data * 1e-3

manifold = PoincareBallExact
optimizer = rsgd.RSGD(model.parameters(), 1e-2, manifold=manifold)
default_gt = torch.zeros(20).long()
criterion = nn.CrossEntropyLoss(reduction="sum")



for i in range(5):
    tloss = 0.
    for x in dataloader:
        optimizer.zero_grad()
        pe = model(x.long())
        ne = model((torch.rand(len(x), 2) * len(dictionary)).long() + 1)
        pd = manifold.distance(pe[:,0,:], pe[:,1,:]).unsqueeze(1)
        nd = manifold.distance(pe[:,0,:].unsqueeze(1).expand_as(ne), ne)

        prediction = -torch.cat((pd, nd), 1)

        loss = criterion(prediction, default_gt[:len(prediction)])
        tloss += loss.item()
        loss.backward()
        optimizer.step()
    print('Loss value for iteration ', i,' is ', tloss)

optimizer = rsgd.RSGD(model.parameters(), 1e-1, manifold=manifold)

for i in range(40):
    tloss = 0.
    for x in dataloader:
        optimizer.zero_grad()
        pe = model(x.long())
        ne = model((torch.rand(len(x), 10) * len(dictionary)).long() + 1)
        pd = manifold.distance(pe[:,0,:], pe[:,1,:]).unsqueeze(1)
        nd = manifold.distance(pe[:,0,:].unsqueeze(1).expand_as(ne), ne)

        prediction = -torch.cat((pd, nd), 1)

        loss = criterion(prediction, default_gt[:len(prediction)])
        tloss += loss.item()
        loss.backward()
        optimizer.step()
    print('Loss value for iteration ', i,' is ', tloss)


from matplotlib import pyplot as plt

plt.figure(figsize=(20, 20))

inverted_dictionary = {v:k for k,v in dictionary.items()}
for x, y in tuple_neigbhor:
    if(model.weight.data[x].norm(2, -1) <  0.991 and model.weight.data[y].norm(2, -1) <  0.991):
        plot_geodesic(model.weight.data[x,:].unsqueeze(0), model.weight.data[y,: ].unsqueeze(0), manifold)
plt.scatter(model.weight.data[1:,0].numpy(), model.weight.data[1:,1].numpy())

for i in range(1, len(model.weight)):
    if(model.weight.data[i].norm(2, -1) <  0.95):

        plt.annotate(inverted_dictionary[i], (model.weight.data[i,0], model.weight.data[i,1]),fontsize=30)
plt.axis('off')
os.makedirs("LOG/hierarchical",  exist_ok=True)
plt.savefig("LOG/hierarchical/"+root_synset+".svg", format="svg")
plt.savefig("LOG/hierarchical/"+root_synset+".png", format="png")
plt.close()