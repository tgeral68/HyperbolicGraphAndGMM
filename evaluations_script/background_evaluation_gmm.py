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

import sys, traceback
import matplotlib.colors as plt_colors
parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')


parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="The folder where to find experiments")        
parser.add_argument('--verbose', dest="verbose", default=False, action="store_true",
                    help="The format of the log")
parser.add_argument('--cuda', dest="cuda", action="store_true") 
parser.add_argument('--init', dest="init", action="store_true")
parser.add_argument('--learned', dest="learned", action="store_true")
parser.add_argument('--force', dest="force", action="store_true")
parser.add_argument('--force-learning', dest="force_learning", action="store_true")
parser.add_argument('--plot', dest="plot", action="store_true")
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
            if(log_prefix+"unsupervised_eval_gmm" not in log_in or args.force):
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
                intras = []
                inters = []
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
                    elif(os.path.isfile(os.path.join(args.folder, directory,"gmm_"+str(i)+".json"))
                            and not args.force_learning):
                        print("File already exists")
                        gmm_saved_weight = torch.load(os.path.join(args.folder, directory,"gmm_"+str(i)+".json"))

                        pi, mu, sigma = (gmm_saved_weight["pi"].to(representations.device),
                                            gmm_saved_weight["mu"].to(representations.device),
                                            gmm_saved_weight["sigma"].to(representations.device)
                                            )
                        current_gmm = PoincareEM(n_gaussian)
                        current_gmm.set_parameters(pi, mu, sigma)
                        gmm_list.append(current_gmm)                 
                    else:
                        algs = PoincareEM(n_gaussian, verbose=False)
                        algs.fit(representations, max_iter=200)
                        gmm_list.append(algs)
                        pi, mu, sigma = algs.get_parameters()
                        save_weight = {"pi": pi, "mu":mu, "sigma":sigma}
                        torch.save(save_weight, os.path.join(args.folder, directory,"gmm_"+str(i)+".json"))


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
                    if(args.plot):
                        colors =[]
                        for i in range(len(D.Y)):
                            # print(prediction[i])
                            colors.append(plt_colors.hsv_to_rgb([prediction[i].item()/(n_gaussian),0.5,0.8]))

                        pi_d, mu_d, sigma_d = current_gmm.get_parameters()
                        print(representations.size())
                        plot_tools.plot_embedding_distribution_multi([representations.cpu()], [pi_d.cpu()], [mu_d.cpu()], [sigma_d.cpu()], 
                                                                    labels=None, N=100, colors=colors, 
                                                                    save_path=os.path.join(args.folder, directory, "gmm_unsupervised.png"))
                        plot_tools.plot_embeddings(representations.cpu(), colors=colors, save_folder=os.path.join(args.folder, directory), file_name="gmm_unsupervised_no_gauss.png")
                    prediction_mat = torch.zeros(len(X), n_gaussian).cuda()
                    gt_mat = [torch.LongTensor(ground_truth[i]).cuda() for i in tqdm.trange(len(D.Y))]
                    print("create prediction matrix")
                    for i in tqdm.trange(len(D.Y)):
                        prediction_mat[i][prediction[i]] = 1
                    print("Evaluate Conductance")
                    conductances.append(evaluation.conductance(prediction_mat, adjency_matrix))
                    nmis.append(evaluation.nmi(prediction_mat, ground_truth))
                    print("Evaluate intra/inter")
                    intras = evaluation.intra_cluster_density(prediction_mat, adjency_matrix)
                    weighted_intra = evaluation.intra_cluster_density(prediction_mat, adjency_matrix, weighted=True)
                    
                # log results
                log_in.append({log_prefix+"unsupervised_eval_gmm": {"accuracy":accuracies,
                                                                    "conductance":conductances,
                                                                    "nmi":nmis,
                                                                    "intra":intras,
                                                                    "w_intra": weighted_intra,
                                                                    "density-total":evaluation.graph_density(X)}})

                import numpy as np
                print("Results of unsupervised evaluation for ", directory)
                print("\t Mean accuracy : ", sum(accuracies, 0.)/args.n)
            else:
                print(directory," already evaluated")
        except :
            traceback.print_exc(file=sys.stdout)
            print("An error occured reading "+directory+" log, this folder will be ignored")
    if args.force:
        args.force = False