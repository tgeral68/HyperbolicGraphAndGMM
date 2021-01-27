'''
This script read the json and print performances obtained regrouping by dimenssion and dataset
'''
import math
import argparse
import torch
import os
import tqdm
from torch.utils.data import DataLoader
import time
from rcome.clustering_tools.poincare_em import PoincareEM
from rcome.clustering_tools.poincare_kmeans import PoincareKMeans
from rcome.data_tools import corpora_tools, corpora, data, config, logger, data_loader
from rcome.function_tools import pytorch_categorical
from rcome.evaluation_tools import evaluation
from rcome.visualisation_tools import plot_tools
from rcome.optim_tools import optimizer
from rcome.community_tools import poincare_classifier as pc

parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')


parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="The folder where to find experiments")        
parser.add_argument('--verbose', dest="verbose", default=False, action="store_true",
                    help="The format of the log")
parser.add_argument('--cuda', dest="cuda", action="store_true") 
parser.add_argument('--init', dest="init", action="store_true")
parser.add_argument('--learned', dest="learned", action="store_true")
parser.add_argument('--precision', dest="precision", type=int, nargs='+', default=[1, 3, 5],
                    help="Precision at  to evaluate (e.g --precision 1 3 5 | evaluate precision at 1, 3 and 5) ") 

parser.add_argument('--n', dest="n", type=int, default=1,
                    help="number of times to perform kmeans")        
args = parser.parse_args()
torch.set_default_tensor_type(torch.DoubleTensor)

list_of_directory = os.listdir(args.folder)
group = {}
log_prefix = 'init-' if(args.init) else ''
while(True):
    time.sleep(10.)
    for directory in list_of_directory:
        try:
            log_in = logger.JSONLogger(os.path.join(args.folder, directory,"log.json"), mod="continue")
            if(log_prefix+"supervised_eval_mlr" not in log_in):
                print('Start evaluate gmm unsupervised for ', directory)
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
                    representations = torch.load(os.path.join(args.folder, directory, "embeddings_init.t7"))
                else:
                    representations = torch.load(os.path.join(args.folder, directory, "embeddings.t7"))[0]

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
                        gmm_saved_weight = torch.load(os.path.join(args.folder, directory,"pi_mu_sigma.t7"))

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
                # log results
                log_in.append({"supervised_evaluation_classifier":scores})
                print("\n\t mean score ->  ",{k:sum(v,0)/5 for k,v in precisions.items()}, "\n\n")

            else:
                print(directory," already evaluated")
        except:
            print("An error occured reading "+directory+" log, this folder will be ignored")
            raise