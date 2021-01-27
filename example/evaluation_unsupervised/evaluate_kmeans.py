import torch

from rcome.evaluation_tools.evaluation import nmi, conductance
from rcome.data_tools import data_loader
from rcome.clustering_tools import poincare_kmeans

# loading data and learned embeddings
X, Y = data_loader.load_corpus("LFR1", directed=False)
n_gaussian = 13
representations = torch.load("LOG/all_loss/representation.pth")
ground_truth = torch.LongTensor([[1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])

# estimate the gmm
kmeans_alg = poincare_kmeans.PoincareKMeans(13)
kmeans_alg.fit(representations)

# predict associated gaussian 
prediction = kmeans_alg.predict(representations)
prediction_mat = torch.LongTensor([[1 if(y in prediction[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])


conductance_scores = conductance(prediction_mat, X)
print("Conductance score is ", torch.Tensor(conductance_scores).mean().item(), "+-",
                            torch.Tensor(conductance_scores).std().item())
nmi_score = nmi(prediction_mat, ground_truth)
print("NMI score is ", nmi_score)