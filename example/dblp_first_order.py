
from torch import nn

from rcome.data_tools import data_loader
from rcome.embedding_tools import losses


nodes, communities = data_loader.load_corpus('dblp', directed=False)
nodes_representation  = nn.Embbedding(len(nodes))

for edges in nodes:
    for l in edges:
        nodes