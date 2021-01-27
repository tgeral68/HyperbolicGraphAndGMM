'''
This script read the json and print performances obtained regrouping by dimenssion and dataset
'''
import math
import argparse
import torch
from rcome.data_tools import logger
import os
parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')


parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="The folder where to find experiments") 
parser.add_argument('--folder-log-substring', dest="fls", type=str, default="DBLP",
                    help="The format of the log")          
parser.add_argument('--verbose', dest="verbose", default=False, action="store_true",
                    help="The format of the log")        
args = parser.parse_args()


list_of_directory = os.listdir(args.folder)
group = {}

for directory in list_of_directory:
    if(args.fls not in directory):
        # list_of_directory.remove(list_of_directory)
        pass
    else:
        try:

            logger_object = logger.JSONLogger(os.path.join(args.folder,directory,"log.json"), mod="continue")
            dct_key = logger_object["dataset"]+"-"+str(logger_object["size"])+"D"
            if(args.verbose):
                print("Folder \""+directory+"\" embedding size->"+str(logger_object["size"]))
            if(dct_key not in group):
                group[dct_key] = {}
            

            # euclidean classifier getter

            ll = logger_object["unsupervised_eval"]["nmi"]
            if(ll[0] == 0.):
                print("0 error in ", directory, " better ignore it please ")
            if("nmi" not in group[dct_key]):
                group[dct_key]["nmi"] = []
            group[dct_key]["nmi"] += ll

            # euclidean classifier getter

            ll = logger_object["unsupervised_eval"]["accuracy"]
            if("accuracy" not in group[dct_key]):
                group[dct_key]["accuracy"] = []
            group[dct_key]["accuracy"] += ll

            ll = logger_object["unsupervised_eval"]["conductance"]
            if("conductance" not in group[dct_key]):
                group[dct_key]["conductance"] = []
            print(ll)
            group[dct_key]["conductance"] += [i if(i!=math.inf) else 1. for i in ll[0]]
        except:
            print("An error occured reading "+directory+" log, this folder will be ignored")

# printing json files

for dct_key in group:
    try:
        print("Results for dataset "+dct_key)
        print("\t NMI")
        v = group[dct_key]["nmi"]
        print("\t\t Results",(str((torch.Tensor(v).mean().item())*1000//1/10)+"+-"+str((torch.Tensor(v).std().item())*1000//1/10)))

        print("\t Precision ")
        v = group[dct_key]["accuracy"]
        print("\t\t Results",(str((torch.Tensor(v).mean().item())*10//1/10)+"+-"+str((torch.Tensor(v).std().item())*10//1/10)))
        print("\t Conductence ")
        v = group[dct_key]["conductance"]
        print("\t\t Results",(str((torch.Tensor(v).mean().item())*1000//1/10)+"+-"+str((torch.Tensor(v).std().item())*1000//1/10)))

    except Exception:
        print("invalid value encountered")
        print()


