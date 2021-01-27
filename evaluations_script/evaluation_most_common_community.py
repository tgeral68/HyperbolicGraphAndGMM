import argparse
import tqdm

import os

import torch
from torch.utils.data import DataLoader

from rcome.visualisation_tools import plot_tools
from rcome.clustering_tools.poincare_em import PoincareEM as EM
from rcome.clustering_tools.poincare_kmeans import PoincareKMeans as KM
from rcome.data_tools import corpora_tools, corpora, data, logger
from rcome.evaluation_tools import evaluation
from rcome.community_tools import poincare_classifier as pc


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--dataset', dest="dataset", type=str, default="dblp",
                    help="The dataset to evaluate") 
parser.add_argument('--cuda', dest="cuda", action="store_true")
parser.add_argument('--n-fold', dest="n_fold", type=int, default=5,
                    help="Number of Cross validation folds") 
parser.add_argument('--precision', dest="precision", type=int, nargs='+', default=[1, 3, 5],
                    help="Precision at  to evaluate (e.g --precision 1 3 5 | evaluate precision at 1, 3 and 5) ") 
args = parser.parse_args()


dataset_dict = { "karate": corpora.load_karate,
            "football": corpora.load_football,
            "flickr": corpora.load_flickr,
            "dblp": corpora.load_dblp,
            "books": corpora.load_books,
            "blogCatalog": corpora.load_blogCatalog,
            "polblog": corpora.load_polblogs,
            "adjnoun": corpora.load_adjnoun,
            "wikipedia": corpora.load_wikipedia
          }

if(args.dataset not in dataset_dict):
    print("Dataset " + args.dataset + " does not exist, please select one of the following : ")
    print(list(dataset_dict.keys()))
    quit()

class MostCommonCommunity():
    def __init__(self, n_communities):
        from collections import Counter
        self.n_communities = n_communities

    def fit(self, X, Y):
        with torch.no_grad():
            # rank labels
            sum_communities = Y.sum(0)
            prob_communities = sum_communities.double()/sum_communities.sum()
            _ , self.indices_ranked = (-prob_communities).sort(0)
            self.prediction_vector = prob_communities
    
    def probs(self, z):

        return self.prediction_vector.unsqueeze(0).expand(z.shape[0], n_communities)*1

    def predict(self, z):

        return self.indices_ranked.unsqueeze(0).expand(z.shape[0], n_communities)*1

D, X, Y = dataset_dict[args.dataset]()
n_communities = max([community for k, y in Y.items() for community in y])
print("Number of communities : ", n_communities)
print("Number of nodes : ", len(X))
ground_truth = torch.LongTensor([[ 1 if(y+1 in Y[i]) else 0 for y in range(n_communities)] for i in range(len(X))])

CVE = evaluation.CrossValEvaluation(torch.rand(ground_truth.shape[0],1), ground_truth, 
                                    nb_set=5, algs_object=MostCommonCommunity)

precisions = {}
for eval_precision in args.precision: 
  precisions["P"+str(eval_precision)] = CVE.get_score(evaluation.PrecisionScore(at=eval_precision))

scores = precisions.copy()
print("\n\t Most Common Community scoring at ->  ",{k:str(sum(v,0)/5*1000//1/10)+"+-"+
                                                    str(torch.tensor(v).std().item()*1000//1/10)
                                                    for k,v in precisions.items()
                                                    }, "\n\n")
mcc = MostCommonCommunity(n_communities)
mcc.fit(torch.rand(ground_truth.shape[0],1), ground_truth)
prediction_probs = mcc.probs(torch.rand(ground_truth.shape[0],1))
precisions = {}
for eval_precision in args.precision: 
  precisions["P"+str(eval_precision)] = evaluation.PrecisionScore(at=eval_precision)(prediction_probs, ground_truth)

print(precisions)
scores = precisions.copy()
print("\n\t Most Common Community(Absolute) scoring at ->  ",{k:str(v*1000//1/10)+"+-"+
                                                            str(torch.tensor(v).std().item()*1000//1/10)
                                                            for k,v in precisions.items()
                                                            }, "\n\n")