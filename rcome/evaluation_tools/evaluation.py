from collections import Counter
import numpy as np
import math
import tqdm
import itertools

import torch

from rcome.function_tools import distribution_function, poincare_function
from rcome.function_tools import euclidean_function as ef

from rcome.clustering_tools.poincare_kmeans import PoincareKMeans
from rcome.clustering_tools.poincare_em import PoincareEM




class EvaluationMetrics(object):
    # both must be matrix 
    def __init__(self, rule_function=None):
        self.rule_function = rule_function

    def __call__(self, prediction, ground_truth):
        self.TP = []
        self.FP = []
        self.TN = []
        self.FN = []
        self.nb_classes = ground_truth.size(-1)
        if(self.rule_function is not None):
            self.prediction = self.rule_function(prediction)

        else:
            self.prediction = prediction

        for i in range(self.nb_classes):
            positive_gt = (ground_truth[:,i] == 1).float()
            positive_pr = (self.prediction[:,i] == 1).float()


            negative_gt = (ground_truth[:,i] == 0).float()
            negative_pr = (self.prediction[:,i] == 0).float()

            tp = (positive_gt * positive_pr).sum()
            fp = (positive_pr * negative_gt).sum()
            self.TP.append(tp.item())
            self.FP.append(fp.item())

            tn = (negative_gt * negative_pr).sum()
            fn = (positive_gt * negative_pr).sum()
            self.TN.append(tn.item())
            self.FN.append(fn.item())
        micro, macro = self.score()
        return {"micro": micro, "macro":macro}

    def micro_precision(self):

        if((sum(self.TP, 0) + sum(self.FP,0)) == 0):
            return 0
        return sum(self.TP, 0)/(sum(self.TP, 0) + sum(self.FP,0))

    def micro_recall(self):
        if((sum(self.TP, 0) + sum(self.FN,0)) == 0 ):
            return 0
        return sum(self.TP, 0)/(sum(self.TP, 0) + sum(self.FN,0))
    
    def micro_F(self):
        m_p, m_r  = self.micro_precision(), self.micro_recall()
        if(m_p + m_r ==0 ):
            return 0
        return (2 * m_p * m_r) /(m_p + m_r)

    def macro_precision(self):
        precision_by_label = [tp/(tp+fp) if(tp+fp != 0) else 0 for tp, fp in zip(self.TP, self.FP)]

        return sum(precision_by_label, 0)/(len(precision_by_label))

    def macro_recall(self):
        recall_by_label = [tp/(tp+fn) if(tp+fn != 0) else 0  for tp, fn in zip(self.TP, self.FN)]
        if(len(recall_by_label) == 0):
            return 0
        return sum(recall_by_label, 0)/(len(recall_by_label))

    def macro_F(self):
        m_p, m_r  = self.macro_precision(), self.macro_recall()
        if(m_p == 0 and m_r == 0):
            return 0
        return (2 * m_p * m_r) /(m_p + m_r)

    def score(self):
        return self.micro_F(), self.macro_F()


def precision_at(prediction, ground_truth, at=5):
    prediction_value, prediction_index = (-prediction).sort(-1)
    trange = torch.arange(len(prediction)).unsqueeze(-1).expand(len(prediction), at).flatten()
    indexes = prediction_index[:,:at].flatten()
    score = ((ground_truth[trange, indexes]).float().view(len(prediction), at)).sum(-1)/at
    return score.mean().item()


def mean_conductance(prediction, adjency_matrix):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    I = {i for i in range(len(prediction))}

    score = 0
    for c in range(K):
        c_nodes = set(prediction[:, c].nonzero().flatten().tolist())
        nc_nodes = I - c_nodes
        cut_score_a = 0
        for i in c_nodes:
            cut_score_a += len(set(adjency_matrix[i]) - c_nodes)
        cut_score_b = 0
        for i in c_nodes:
            cut_score_b += len(adjency_matrix[i])

        cut_score_c = 0
        for i in nc_nodes:
            cut_score_c += len(adjency_matrix[i])
        if(cut_score_b==0 or cut_score_c ==0):
            score += 0 
        else:
            score += cut_score_a/(min(cut_score_b, cut_score_c))
    
    return score/K

def conductance(prediction, adjency_matrix):
    import math
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    # print(K)
    I = {i for i in range(len(prediction))}
    conductances = []
    for c in range(K):
        c_nodes = set(torch.nonzero(prediction[:, c]).flatten().tolist())
        nc_nodes = I - c_nodes
        cut_score_a = 0
        for i in c_nodes:
            cut_score_a += len(set(adjency_matrix[i]) - c_nodes)
        cut_score_b = 0
        for i in c_nodes:
            cut_score_b += len(adjency_matrix[i])

        cut_score_c = 0
        for i in nc_nodes:
            cut_score_c += len(adjency_matrix[i])
        if(cut_score_b==0 or cut_score_c ==0):
            conductances.append(math.inf)
        else:
            conductances.append(cut_score_a/(min(cut_score_b, cut_score_c)))

    return conductances

def nmi(prediction, ground_truth):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)

    I = {i for i in range(len(prediction))}

    PN = []
    GN = []
    den = 0
    for i in range(K):
        PN.append(set(torch.nonzero(prediction[:, i]).flatten().tolist()))
        GN.append(set(torch.nonzero(ground_truth[:, i]).flatten().tolist()))
        if(len(PN[-1]) != 0):
            den += len(PN[-1]) * math.log(len(PN[-1])/N)
        if(len(GN[-1]) != 0):
            den += len(GN[-1]) * math.log(len(GN[-1])/N)
    num = 0
    for a in PN:
        for b in GN:
            N_ij = len(a.intersection(b))
            if(N_ij != 0):
                num += N_ij * math.log((N_ij * N)/(len(a) *len(b) ))
    
    return -2 * (num/den)



@DeprecationWarning
def intra_cluster_density(annotation, adjency_matrix, weighted=False):
    N, K = annotation.shape[0], annotation.shape[-1]
    n_relevant_clusters = 0.
    kappa_intra = 0.
    for cluster_index in range(K):
        cluster_kappa_intra = 0.
        # getting nodes of the cluster and size of cluster
        cluster_nodes = annotation[:, cluster_index].nonzero().flatten().tolist()
        cluster_size = len(cluster_nodes)
        # if a cluster do not contain one elts (wikipedia)
        if(cluster_size > 1):
            for i, node_source in enumerate(cluster_nodes):
                for j, node_target in enumerate(cluster_nodes[i+1:]):
                    if(node_target in adjency_matrix[node_source]):
                        cluster_kappa_intra += 1.
            cluster_score = cluster_kappa_intra / ( 0.5 * cluster_size * ( cluster_size - 1 ) ) 
            if(weighted):
                cluster_score *= cluster_size
            kappa_intra += cluster_score
            n_relevant_clusters += 1.
    return kappa_intra / n_relevant_clusters

@DeprecationWarning
def graph_density(adjency_matrix):
    return float(sum([len(v) for k,v in adjency_matrix.items()], 0))/( len(adjency_matrix) * (len(adjency_matrix)-1))

@DeprecationWarning
def intra_density(prediction, adjency_matrix):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    I = {i for i in range(len(prediction))}
    # print(adjency_matrix.shape)
    print(K)
    score = 0
    for c in range(K):
        c_nodes = set(prediction[:, c].nonzero().flatten().tolist())
        n_intra_edges = 0
        for i in c_nodes:
            for j in c_nodes:
                if(j in adjency_matrix[i]):
                    n_intra_edges += 1
        print(n_intra_edges, len(c_nodes))
        score += n_intra_edges/( len(c_nodes) * (len(c_nodes)-1))
    return score/K
@DeprecationWarning
def inter_density(prediction, adjency_matrix):
    N = prediction.size(0)
    # the number of clusters
    K = prediction.size(-1)
    I = {i for i in range(len(prediction))}

    score = 0
    for c in range(K):
        c_nodes = set(prediction[:, c].nonzero().flatten().tolist())

        n_inter_edges = 0
        for c_o in range(c+1, K):
            c_nodes_o = set(prediction[:, c_o].nonzero().flatten().tolist())
            for i in c_nodes:
                for j in c_nodes_o:
                    if(j in adjency_matrix[i]):
                        n_inter_edges += 1
            score += (n_inter_edges)/(len(c_nodes) * len(c_nodes_o))
    return score/(K * (K - 1))

class PrecisionScore(object):
    def __init__(self, at=5):
        self.at = at

    def __call__(self, x, y):
        return precision_at(x, y, at=self.at)




class CrossValEvaluation(object):
    def __init__(self, embeddings, ground_truth, nb_set=5, algs_object=PoincareEM):
        self.algs_object = algs_object
        self.z = embeddings
        self.gt = ground_truth
        self.nb_set = nb_set
        # split set
        subset_index = torch.randperm(len(self.z))
        nb_value = len(self.z)//nb_set
        self.subset_indexer = [subset_index[nb_value *i:min(nb_value * (i+1), len(self.z))] for i in range(nb_set)]
        self.all_algs = []
        pb = tqdm.trange(len(self.subset_indexer))

        for i, test_index in zip(pb, self.subset_indexer):
            # create train dataset being concatenation of not current test set
            train_index = torch.cat([ subset for ci, subset in enumerate(self.subset_indexer) if(i!=ci)], 0)
            
            # get embeddings sets
            train_embeddings = self.z[train_index]
            test_embeddings  = self.z[test_index]

            # get ground truth sets
            print("GT SIZE ", self.gt.size())
            train_labels = self.gt[train_index]
            test_labels  =  self.gt[test_index]
            algs = self.algs_object(self.gt.size(-1))
            algs.fit(train_embeddings, Y=train_labels)
            self.all_algs.append(algs)
        
    def get_score(self, scoring_function):
        scores = []
        pb = tqdm.trange(len(self.subset_indexer))
        for i, test_index in zip(pb, self.subset_indexer):
            # create train dataset being concatenation of not current test set
            train_index = torch.cat([ subset for ci, subset in enumerate(self.subset_indexer) if(i!=ci)], 0)
            
            # get embeddings sets
            train_embeddings = self.z[train_index]
            test_embeddings  = self.z[test_index]

            # get ground truth sets
            train_labels = self.gt[train_index]
            test_labels  =  self.gt[test_index]

            prediction = self.all_algs[i].probs(test_embeddings)

            set_score = scoring_function(prediction, test_labels)
            scores.append(set_score)
        return scores


def accuracy(prediction, labels):
    return (prediction == labels).float().mean()

@DeprecationWarning
def precision1_unsupervised_evaluation(cluster_prediction, ground_truth, n_distrib):
    print("n_distrib ", n_distrib)
    print("min gt label estimation ",torch.LongTensor([ground_truth[i][0] for i in range(len(ground_truth))]).min())
    labels_matrix = torch.zeros(n_distrib, n_distrib).cuda()
    cluster_prediction = cluster_prediction.cuda()
    import tqdm
    gt_matrix = [torch.LongTensor(ground_truth[i]).cuda() for i in tqdm.trange(len(ground_truth))]

    for i in tqdm.trange(len(ground_truth)):
        labels_matrix[cluster_prediction[i].item()][gt_matrix[i]] += 1


    permutation = torch.arange(0, n_distrib)
    for i in tqdm.trange(n_distrib):
        values_cluster, index_cluster = labels_matrix.max(0)
        index_gt = values_cluster.argmax(0).item()
        max_index_x, max_index_y = index_cluster[index_gt], index_gt
        permutation[max_index_x] = max_index_y
        labels_matrix[max_index_x] = -1
        labels_matrix[:, max_index_y] = -1
    
    precision_at_1 = 0.
    for i in tqdm.trange(len(cluster_prediction)):
        if(permutation[cluster_prediction[i]].item() in  ground_truth[i]):
            precision_at_1 += 1
    print("ici")
    return precision_at_1/len(cluster_prediction)
