import argparse
import tqdm
import os

import torch
from torch.utils.data import DataLoader

from rcome.visualisation_tools import plot_tools
from rcome.clustering_tools.poincare_em import PoincareEM as EM
from rcome.clustering_tools.poincare_kmeans import PoincareKMeans as KM
from rcome.data_tools import corpora_tools, corpora, data, logger, config, data_loader
from rcome.evaluation_tools import evaluation
from rcome.community_tools import poincare_classifier as pc


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="embeddings location folder") 
parser.add_argument('--id', dest="id", type=str, default="",
                    help="id of the experiment to evaluate") 
parser.add_argument('--init', dest="init", action="store_true") 
parser.add_argument('--cuda', dest="cuda", action="store_true") 
parser.add_argument('--n-fold', dest="n_fold", type=int, default=5,
                    help="Number of Cross validation folds") 
parser.add_argument('--precision', dest="precision", type=int, nargs='+', default=[1, 3, 5],
                    help="Precision at  to evaluate (e.g --precision 1 3 5 | evaluate precision at 1, 3 and 5) ") 
args = parser.parse_args()



if(args.folder == "" and args.id == ""):
  print("Please give arguments for --folder or --id")
  quit()
if args.folder == "" and args.id != "":
  global_config = config.ConfigurationFile("./DATA/config.conf")
  saving_folder = global_config["save_folder"]
  args.folder = os.path.join(saving_folder, args.id)

log_in = logger.JSONLogger(os.path.join(args.folder,"log.json"), mod="continue")
dataset_name = log_in["dataset"]
torch.set_default_tensor_type(torch.DoubleTensor)

n_gaussian = log_in["n_gaussian"]
print("EVALUATE SUPERVISED CLUSTERING ON ")
print("Dataset -> ",dataset_name)
print("Number of communities -> ", n_gaussian)
size = log_in["size"]

print("\tLoading Corpus ")
X, Y = data_loader.load_corpus(dataset_name)
D = corpora.RandomWalkCorpus(X, Y)
if(args.init):
  print("init embedding")
  representations = torch.load(os.path.join(args.folder,"embeddings_init.t7"))
  prep = "init-"
else:
  representations = torch.load(os.path.join(args.folder,"embeddings.t7"))[0]
  prep = ""

if(args.cuda):
  representations = representations.cuda()
print("\trep -> ", representations.size())
ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
if(args.cuda):
  ground_truth = ground_truth.cuda()
print("\tGround truth size ", ground_truth.size())
print(ground_truth.sum(0))

print("##########################GMM Hyperbolic###############################")

CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=args.n_fold, algs_object=EM)
precisions = {}
for eval_precision in args.precision: 
  precisions["P"+str(eval_precision)] = CVE.get_score(evaluation.PrecisionScore(at=eval_precision))

scores = precisions


print("\n\t mean score EM ->  ",{k:sum(v,0)/5 for k,v in precisions.items()}, "\n\n")

log_in.append({prep+"supervised_evaluation_gmm":scores})
if(size == 2):
    import matplotlib.pyplot as plt
    import matplotlib.colors as plt_colors
    import numpy as np
    unique_label = np.unique(sum([ y for k, y in D.Y.items()],[]))
    colors = []

    for i in range(len(D.Y)):
        colors.append(plt_colors.hsv_to_rgb([D.Y[i][0]/(len(unique_label)),0.5,0.8]))

    gmm = CVE.all_algs[0]
    pi_d, mu_d, sigma_d = gmm.get_parameters()
    print(representations.size())
    plot_tools.plot_embedding_distribution_multi([representations.cpu()], [pi_d.cpu()], [mu_d.cpu()], [sigma_d.cpu()], 
                                                labels=None, N=100, colors=colors, 
                                                save_path=os.path.join(args.folder, "gmm_supervised.png"))

CVE = evaluation.CrossValEvaluation(representations, ground_truth,  nb_set=args.n_fold, algs_object=KM)
precisions = {}
for eval_precision in args.precision: 
  precisions["P"+str(eval_precision)] = CVE.get_score(evaluation.PrecisionScore(at=eval_precision))

scores = precisions


print("\n\t mean score kmean ->  ",{k:sum(v,0)/5 for k,v in precisions.items()}, "\n\n")

log_in.append({prep+"supervised_evaluation_kmean":scores})