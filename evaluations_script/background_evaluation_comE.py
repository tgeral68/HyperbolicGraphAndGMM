
import torch
import argparse
import tqdm
import io
import os

import time
from rcome.clustering_tools.euclidean_em import GMM, GaussianMixtureSKLearn, GMMSpherical
from rcome.community_tools.euclidean_classifier import EuclideanClassifier, SKLearnSVM
from rcome.clustering_tools.euclidean_kmeans import KMeans
from rcome.data_tools import corpora_tools, corpora, data, logger, data_loader
from rcome.evaluation_tools import evaluation


parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')

parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="Experiments folder file") 
parser.add_argument('--force', dest="force", action="store_true", default=False,
                    help="Do force relaunch experiment evaluation") 
parser.add_argument('--covariance-type', dest="covariance_type", type=str, default="full",
                    help="Type of Gaussian covariance (using sklearn: full(default), tied, diag, spherical)") 
args = parser.parse_args()

nb_communities_dict= {
    "dblp":5,
    "flickr":195,
    "blogCatalog":39,
    "wikipedia":40,
    "LFR1":13,
    "LFR2":13,
    "LFR3":13,
    "LFR4":13,
    "LFR5":13,
    "LFR6":13,
    "LFR7":13,
    "LFR8":13,
    "LFR9":13
}



while(True):
    time.sleep(10.)
    list_of_directory = os.listdir(args.folder)
    for directory in list_of_directory:
        try:
            path_to_experiment = os.path.join(args.folder,directory)
            logger_object = logger.JSONLogger(os.path.join(path_to_experiment,"log.json"), mod="continue")
            # if the experiment has already been evaluated we go to the next one
            if("supervised_eval" in logger_object and "supervised_eval"  in logger_object and not args.force):
                print(directory, "  has already been evaluated")
                continue
            else:
                pass
            
            if(0 == 1):
                pass
            else:
                print(directory)
                # set info vars
                print('Loading Corpus')
                dataset_name = logger_object["dataset"]
                n_communities = nb_communities_dict[dataset_name]

                # loading corpus and representation 
                X, Y = data_loader.load_corpus(dataset_name, directed=True)
                print("Size ", len(X))
                D = corpora.RandomWalkCorpus(X, Y)
                with io.open(os.path.join(path_to_experiment, "embeddings.txt")) as embedding_file:
                    V = []
                    for line in embedding_file:
                        splitted_line = line.split()
                        V.append([float(splitted_line[i+1]) for i in range(len(splitted_line)-1)])
                representations = torch.Tensor(V)


                #### unsupervised evaluation ####

                # fitting GMM
                algs = GaussianMixtureSKLearn(n_communities, covariance_type=args.covariance_type)
                algs.fit(representations)
                
                #get and transform prediction
                prediction = algs.predict(representations).long()
                prediction_mat = torch.LongTensor([[ 1 if(y == prediction[i]) else 0 for y in range(n_communities)] 
                                                    for i in range(len(X))])
                
                # get the conductance
                conductance = evaluation.conductance(prediction_mat, X)
                
                # get the NMI
                ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_communities)] 
                                                for i in range(len(X))])
                nmi = evaluation.nmi(prediction_mat, ground_truth)
                print("Evaluate intra/inter")
                intras = evaluation.intra_cluster_density(prediction_mat, X)
                # inters = evaluation.inter_density(prediction_mat, X)

                # get the accuracy
                accuracy = evaluation.precision1_unsupervised_evaluation(prediction, D.Y, n_communities)

                ## fill results
                results_unsupervised = {"accuracy":accuracy, "conductance":conductance, "nmi":nmi,
                                        "intra": intras,  "density-total":evaluation.graph_density(X)}
                print(results_unsupervised)
                logger_object.append({"unsupervised_eval": results_unsupervised})


                # #### supervised evaluation ####
                ground_truth = torch.LongTensor([[ 1 if(y in Y[i]) else 0 for y in range(n_communities)] for i in range(len(X))])

                CVE = evaluation.CrossValEvaluation(representations, ground_truth, nb_set=5, algs_object=EuclideanClassifier)

                def threshold_rule(prediction_tensor):

                    prediction_tensor_copy = prediction_tensor.clone()
                    prediction_tensor[torch.arange(len(prediction_tensor)),prediction_tensor.max(-1)[1]] =1
                    prediction_tensor_copy[prediction_tensor>=0.5] = 1
                    prediction_tensor_copy[prediction_tensor<0.5] = 0
                    return prediction_tensor_copy

                scoring_function = evaluation.EvaluationMetrics(rule_function=threshold_rule)
                macro_micro = CVE.get_score(scoring_function)

                p1 = CVE.get_score(evaluation.PrecisionScore(at=1))

                p3 = CVE.get_score(evaluation.PrecisionScore(at=3))

                p5 = CVE.get_score(evaluation.PrecisionScore(at=5))
                scores = {"macro_micro":macro_micro,"P1":p1, "P3":p3, "P5":p5}

                scores = {"P1":p1, "P3":p3, "P5":p5}

                logger_object.append({"supervised_eval":{"linear_logit":scores}})


        except Exception as e:
            print("ERROR reading -> ", directory)
            print(e)