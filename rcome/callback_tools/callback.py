import torch

from rcome.clustering_tools import poincare_kmeans  as pkm
from rcome.clustering_tools import poincare_em as pem
from rcome.evaluation_tools import evaluation

def log_callback_kmeans_conductance(embeddings, adjancy_matrix, n_centroid):
    kmeans = pkm.PoincareKMeans(n_centroid)
    kmeans.fit(embeddings)
    i =  kmeans.predict(embeddings)
    r = torch.arange(0, i.size(0), device=i.device)
    prediction = torch.zeros(embeddings.size(0), n_centroid)
    prediction[r,i] = 1
    return {"conductance":evaluation.mean_conductance(prediction, adjancy_matrix)}

def log_callback_em_conductance(embeddings, adjancy_matrix, n_centroid):
    kmeans = pem.PoincareEM(n_centroid)
    kmeans.fit(embeddings)
    i =  kmeans.predict(embeddings)
    r = torch.arange(0, i.size(0), device=i.device)
    prediction = torch.zeros(embeddings.size(0), n_centroid)
    prediction[r,i] = 1
    return {"conductance":evaluation.mean_conductance(prediction, adjancy_matrix)}