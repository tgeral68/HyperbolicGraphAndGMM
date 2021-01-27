import argparse
import tqdm
import os

import torch
from torch.utils.data import DataLoader

from rcome.clustering_tools.poincare_em import PoincareEM as EM
from rcome.data_tools import corpora_tools, corpora, data, logger, config, data_loader
from rcome.evaluation_tools import evaluation
from rcome.community_tools import poincare_classifier as pc


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="embeddings location folder") 
parser.add_argument('--id', dest="id", type=str, default="",
                    help="id of the experiment to evaluate") 
parser.add_argument('--init', dest="init", action="store_true") 
parser.add_argument('--precision', dest="precision", type=int, nargs='+', default=[1, 3, 5],
                    help="Precision at  to evaluate (e.g --precision 1 3 5 | evaluate precision at 1, 3 and 5) ") 
parser.add_argument('--cuda', dest="cuda", action="store_true")
args = parser.parse_args()


torch.set_default_tensor_type(torch.DoubleTensor)
if(args.folder == "" and args.id == ""):
  print("Please give arguments for --folder or --id")
  quit()
if args.folder == "" and args.id != "":
  global_config = config.ConfigurationFile("./DATA/config.conf")
  saving_folder = global_config["save_folder"]
  args.folder = os.path.join(saving_folder,  args.id)

log_in = logger.JSONLogger(os.path.join(args.folder,"log.json"), mod="continue")
dataset_name = log_in["dataset"]
n_gaussian = log_in["n_gaussian"]
print("EVALUATE SUPERVISED CLUSTERING ON ")
print("\t Dataset -> ",dataset_name)
print("\t Number of communities -> ", n_gaussian)
size = log_in["size"]


# print("Loading Corpus ")
X, Y = data_loader.load_corpus(dataset_name)

representations_init = torch.load(os.path.join(args.folder,"embeddings_init.t7"))

representations = torch.load(os.path.join(args.folder,"embeddings.t7"))[0]


print("\t Representation matrix shape -> ", representations.size())
ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_gaussian)] for i in range(len(X))])
print("\t Distribution of communities -> ", ground_truth.sum(0))
if(args.cuda):
  representations = representations.cuda()
  representations_init = representations_init.cuda()
  ground_truth = ground_truth.cuda()
CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=pc.PoincareClassifier)
def threshold_rule(prediction_tensor):
  prediction_tensor_copy = prediction_tensor.clone()
  prediction_tensor_copy[torch.arange(len(prediction_tensor)),prediction_tensor.max(-1)[1]] =1
  prediction_tensor_copy[prediction_tensor>=0.5] = 1
  prediction_tensor_copy[prediction_tensor<0.5] = 0
  return prediction_tensor_copy

scoring_function = evaluation.EvaluationMetrics(rule_function=threshold_rule)
macro_micro = CVE.get_score(scoring_function)

precisions = {}
for eval_precision in args.precision: 
  precisions["P"+str(eval_precision)] = CVE.get_score(evaluation.PrecisionScore(at=eval_precision))

scores = precisions.copy()
scores.update({"macro_micro":macro_micro})


print("\n\t mean score ->  ",{k:sum(v,0)/5 for k,v in precisions.items()}, "\n\n")


log_in.append({"supervised_evaluation_classifier":scores})

CVE = evaluation.CrossValEvaluation(representations_init, ground_truth, nb_set=5, algs_object=pc.PoincareClassifier)
scoring_function = evaluation.EvaluationMetrics(rule_function=threshold_rule)
macro_micro = CVE.get_score(scoring_function)

precisions = {}
for eval_precision in args.precision: 
  precisions["P"+str(eval_precision)] = CVE.get_score(evaluation.PrecisionScore(at=eval_precision))

scores = precisions.copy()
scores.update({"macro_micro":macro_micro})



print("\n\t mean score init ->  ",{k:sum(v,0)/5 for k,v in precisions.items()}, "\n\n")

log_in.append({"supervised_evaluation_classifier_init":scores})

