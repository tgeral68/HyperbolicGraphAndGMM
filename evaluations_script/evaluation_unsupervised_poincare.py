import argparse
import tqdm
import os

import torch
from torch.utils.data import DataLoader

from rcome.clustering_tools.poincare_em import PoincareEM
from rcome.clustering_tools.poincare_kmeans import PoincareKMeans
from rcome.data_tools import corpora_tools, corpora, data, config, logger, data_loader
from rcome.function_tools import pytorch_categorical
from rcome.evaluation_tools import evaluation
from rcome.visualisation_tools import plot_tools
from rcome.optim_tools import optimizer


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="embeddings location folder")
parser.add_argument('--id', dest="id", type=str, default="",
                    help="id of the experiment to evaluate") 
parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")          
parser.add_argument('--init', dest="init", action="store_true") 
parser.add_argument('--cuda', dest="cuda", action="store_true") 
parser.add_argument('--verbose', dest="verbose", action="store_true") 


args = parser.parse_args()


if(args.folder == "" and args.id == ""):
  print("Please give arguments for --folder or --id")
  quit()
if args.folder == "" and args.id != "":
  global_config = config.ConfigurationFile("./DATA/config.conf")
  saving_folder = global_config["save_folder"]
  args.folder = os.path.join(saving_folder, args.id)

log_in = logger.JSONLogger(os.path.join(args.folder,"log.json"), mod="continue")

torch.set_default_tensor_type(torch.DoubleTensor)

dataset_name = log_in["dataset"]
print("Dataset : ", dataset_name)
n_gaussian = log_in["n_gaussian"]
dim = log_in["size"]

log_prefix = 'init-' if(args.init) else ''
print("Loading Corpus ", dataset_name)

X, Y = data_loader.load_corpus(dataset_name)
D = corpora.RandomWalkCorpus(X, Y)
print("Dataset statistic ")
print("\tn_nodes = ", len(X))
print("\tn_edges = ", sum([len(y) for y in X.values()], 0))
print("\tn_communities = ", len({c for y in Y.values() for c in y}))
results = []

if(args.init):
  print("init embedding")
  representations = torch.load(os.path.join(args.folder,"embeddings_init.t7"))
else:
  representations = torch.load(os.path.join(args.folder,"embeddings.t7"))[0]

# if gpu evaluation
if args.cuda:
  representations = representations.cuda()

gmm_list = []

accuracies = []
conductances = []
nmis = []

if args.verbose :
  print("Processing expectation-maximisation for gaussian mixture model")
# fitting gmm using em algorithm
for i in tqdm.trange(args.n):
    algs = PoincareEM(n_gaussian, verbose=False)
    algs.fit(representations)
    gmm_list.append(algs)

adjency_matrix = X
ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
if args.verbose :
  print("\nEvaluate accuracy, conductence and normalised mutual information")
for i in range(args.n):
    current_gmm = gmm_list[i]
    accuracies.append(evaluation.poincare_unsupervised_em(representations, D.Y, n_gaussian,
                                                          em=current_gmm,  verbose=False))
    prediction = current_gmm.predict(representations)
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
    conductances.append(evaluation.conductance(prediction_mat, adjency_matrix))
    nmis.append(evaluation.nmi(prediction_mat, ground_truth))

# log results
log_in.append({log_prefix+"unsupervised_eval": {"accuracy":accuracies, "conductance":conductances, "nmi":nmis}})



import numpy as np
# print results
print("Results of unsupervised evaluation ")
print("\t Mean accuracy : ", sum(accuracies, 0.)/args.n)
print("\t Mean conductence : ", sum([sum(v, 0.)/len(v) for v in conductances], 0.)/args.n)
print("\t Top1 conductence : ", sum([np.sort(v)[-1] for v in conductances], 0.)/args.n)
print("\t Top2 conductence : ", sum([np.sort(v)[-2] for v in conductances], 0.)/args.n)
print("\t Min conductence : ", sum([np.sort(v)[0] for v in conductances], 0.)/args.n)
print("\t Mean nmi : ", sum(nmis, 0.)/args.n)

gmm_saved_weight = torch.load(os.path.join(args.folder,"pi_mu_sigma.t7"))

pi, mu, sigma = (gmm_saved_weight["pi"].to(representations.device),
                 gmm_saved_weight["mu"].to(representations.device),
                 gmm_saved_weight["sigma"].to(representations.device)
                )
current_gmm = PoincareEM(n_gaussian)
current_gmm.set_parameters(pi, mu, sigma)
accuracy = evaluation.poincare_unsupervised_em(representations, D.Y, n_gaussian,
                                                      em=current_gmm,  verbose=False)
prediction = current_gmm.predict(representations)
prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
conductance = evaluation.conductance(prediction_mat, adjency_matrix)
nmi = evaluation.nmi(prediction_mat, ground_truth)

# log results
log_in.append({log_prefix+"unsupervised_eval_saved_weight": {"accuracy":accuracy, "conductance":conductance, "nmi":nmi}})

print("Results of unsupervised evaluation saved weight ")
print("\t Mean accuracy : ", accuracy)
print("\t Mean conductence : ", sum(conductance, 0.)/len(conductance))
print("\t Top1 conductence : ", np.sort(conductance)[-1])
print("\t Top2 conductence : ", np.sort(conductance)[-2])
print("\t Min conductence : ", np.sort(conductance)[0])
print("\t Mean nmi : ", nmi)



gmm_list = []

accuracies = []
conductances = []
nmis = []

if args.verbose :
  print("Processing expectation-maximisation for gaussian mixture model")
# fitting gmm using em algorithm
for i in tqdm.trange(args.n):
    algs = PoincareKMeans(n_gaussian, verbose=False)
    algs.fit(representations)
    gmm_list.append(algs)

adjency_matrix = X
ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
if args.verbose :
  print("\nEvaluate accuracy, conductence and normalised mutual information")
for i in range(args.n):
    current_gmm = gmm_list[i]
    accuracies.append(evaluation.poincare_unsupervised_em(representations, D.Y, n_gaussian,
                                                          em=current_gmm,  verbose=False))
    prediction = current_gmm.predict(representations)
    prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
    conductances.append(evaluation.conductance(prediction_mat, adjency_matrix))
    nmis.append(evaluation.nmi(prediction_mat, ground_truth))

# log results
log_in.append({log_prefix+"unsupervised_eval_kmeans": {"accuracy":accuracies, "conductance":conductances, "nmi":nmis}})



import numpy as np
# print results
print("Results of unsupervised evaluation ")
print("\t Mean accuracy : ", sum(accuracies, 0.)/args.n)
print("\t Mean conductence : ", sum([sum(v, 0.)/len(v) for v in conductances], 0.)/args.n)
print("\t Top1 conductence : ", sum([np.sort(v)[-1] for v in conductances], 0.)/args.n)
print("\t Top2 conductence : ", sum([np.sort(v)[-2] for v in conductances], 0.)/args.n)
print("\t Min conductence : ", sum([np.sort(v)[0] for v in conductances], 0.)/args.n)
print("\t Mean nmi : ", sum(nmis, 0.)/args.n)
