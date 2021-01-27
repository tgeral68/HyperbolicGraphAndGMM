import torch

from rcome.evaluation_tools.evaluation import PrecisionScore,  CrossValEvaluation
from rcome.data_tools import data_loader
from rcome.clustering_tools import poincare_em

# loading data and learned embeddings
X, Y = data_loader.load_corpus("LFR1", directed=False)
n_gaussian = 13
representations = torch.load("LOG/all_loss/representation.pth")
ground_truth = torch.LongTensor([[1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])


CVE = CrossValEvaluation(representations, ground_truth, algs_object=poincare_em.PoincareEM)
score = CVE.get_score(PrecisionScore(at=1))
print("Mean precision at 1 (accuracy) ", torch.Tensor(score).mean().item(),
      " +- ", torch.Tensor(score).std().item())