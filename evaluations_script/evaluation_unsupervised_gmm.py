import math
import argparse
import torch
import os
import tqdm
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
parser.add_argument('--learned', dest="learned", action="store_true")
parser.add_argument('--verbose', dest="verbose", action="store_true") 
args = parser.parse_args()
torch.set_default_tensor_type(torch.DoubleTensor)
if(args.folder == "" and args.id == ""):
  print("Please give arguments for --folder or --id")
  quit()
if args.folder == "" and args.id != "":
  global_config = config.ConfigurationFile("./DATA/config.conf")
  saving_folder = global_config["save_folder"]
  args.folder = os.path.join(saving_folder, args.id)

log_in = logger.JSONLogger(os.path.join(args.folder,"log.json"), mod="continue")

if("unsupervised_eval_gmm" not in log_in):
    print('Start evaluate gmm unsupervised for ', args.folder)
torch.set_default_tensor_type(torch.DoubleTensor)

dataset_name = log_in["dataset"]
print("Dataset : ", dataset_name)
n_gaussian = log_in["n_gaussian"]
dim = log_in["size"]


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
    representations = torch.load(os.path.join(args.folder, "embeddings_init.t7"))
else:
    representations = torch.load(os.path.join(args.folder, "embeddings.t7"))[0]

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
    if(args.learned):
        print("loading weight")
        gmm_saved_weight = torch.load(os.path.join(args.folder, "pi_mu_sigma.t7"))

        pi, mu, sigma = (gmm_saved_weight["pi"].to(representations.device),
                        gmm_saved_weight["mu"].to(representations.device),
                        gmm_saved_weight["sigma"].to(representations.device)
                        )
        current_gmm = PoincareEM(n_gaussian)
        current_gmm.set_parameters(pi, mu, sigma)
        gmm_list.append(current_gmm)
    else:
        algs = PoincareEM(n_gaussian, verbose=False)
        algs.fit(representations, max_iter=50)
        gmm_list.append(algs)
print("start evaluation")
adjency_matrix = X
ground_truth = torch.LongTensor([[1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
if args.verbose :
    print("\nEvaluate accuracy, conductence and normalised mutual information")
import tqdm
for i in range(args.n):
    current_gmm = gmm_list[i]
    prediction = current_gmm.predict(representations)
    accuracies.append(evaluation.precision1_unsupervised_evaluation(prediction, D.Y,
                                                                    n_gaussian,
                                                                    )
                        )

    prediction_mat = torch.zeros(len(X), n_gaussian).cuda()
    gt_mat = [torch.LongTensor(ground_truth[i]).cuda() for i in tqdm.trange(len(D.Y))]
    print("create prediction matrix")
    for i in tqdm.trange(len(D.Y)):
        prediction_mat[i][prediction[i]] = 1
    print("Evaluate Conductance")
    conductances.append(evaluation.conductance(prediction_mat, adjency_matrix))
    nmis.append(evaluation.nmi(prediction_mat, ground_truth))

# log results
log_in.append({"unsupervised_eval_gmm": {"accuracy":accuracies, "conductance":conductances, "nmi":nmis}})

import numpy as np
print("Results of unsupervised evaluation for ", args.folder)
print("\t Mean accuracy : ", sum(accuracies, 0.)/args.n)