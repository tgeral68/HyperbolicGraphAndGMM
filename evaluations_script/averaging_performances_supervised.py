'''
This script read the json and print performances obtained regrouping by dimenssion and dataset
'''
import argparse
import torch
from rcome.data_tools import logger
import os
parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')


parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="The folder where to find experiments") 
parser.add_argument('--folder-log-substring', dest="fls", type=str, default=".*",
                    help="The format of the log")          
parser.add_argument('--verbose', dest="verbose", default=False, action="store_true",
                    help="The format of the log")        
args = parser.parse_args()


list_of_directory = os.listdir(args.folder)
group = {}

for directory in list_of_directory:
    if(args.fls not in directory and args.fls != ".*"):
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
            
            ll = logger_object["supervised_evaluation_classifier"]
            if("supervised_classifier" not in group[dct_key]):
                group[dct_key]["supervised_classifier"]={"P1":[], "P3":[], "P5":[]}
            group[dct_key]["supervised_classifier"]["P1"] += ll["P1"]
            if("P3" in ll):
                group[dct_key]["supervised_classifier"]["P3"] += ll["P3"]
                group[dct_key]["supervised_classifier"]["P5"] += ll["P5"]

            # euclidean classifier getter
            
            ll = logger_object["supervised_evaluation_kmean"]
            if("supervised_kmeans" not in group[dct_key]):
                group[dct_key]["supervised_kmeans"]={"P1":[], "P3":[], "P5":[]}
            group[dct_key]["supervised_kmeans"]["P1"] += ll["P1"]
            if("P3" in ll):
                group[dct_key]["supervised_kmeans"]["P3"] += ll["P3"]
                group[dct_key]["supervised_kmeans"]["P5"] += ll["P5"]

            ll = logger_object["supervised_evaluation_gmm"]
            if("supervised_gmm" not in group[dct_key]):
                group[dct_key]["supervised_gmm"]={"P1":[], "P3":[], "P5":[]}
            group[dct_key]["supervised_gmm"]["P1"] += ll["P1"]
            if("P3" in ll):
                group[dct_key]["supervised_gmm"]["P3"] += ll["P3"]
                group[dct_key]["supervised_gmm"]["P5"] += ll["P5"]
        except:
            print("An error occured reading "+directory+" log, this folder will be ignored")

# printing json files

for dct_key in group:
    try:
        print("Results for dataset "+dct_key)
        print("NB experiments ", len(group[dct_key]["supervised_classifier"]["P1"]))
        print("\t\t Results",{k:(str((torch.Tensor(v).mean().item())*1000//1/10)+"+-"+str((torch.Tensor(v).std().item())*1000//1/10))
                                    for k,v in group[dct_key]["supervised_classifier"].items()})

        print("\t supervised GMM ")
        # print(group[dct_key])
        print("\t\t Results",{k:(str((torch.Tensor(v).mean().item())*1000//1/10)+"+-"+str((torch.Tensor(v).std().item())*1000//1/10)) 
                                    for k,v in group[dct_key]["supervised_gmm"].items()})
        print("\t supervised kmeans ")
        print("\t\t Results",{k:str((torch.Tensor(v).mean().item())*1000//1/10)+"+-"+str((torch.Tensor(v).std().item())*1000//1/10) 
                                    for k,v in group[dct_key]["supervised_kmeans"].items()})

    except Exception:
        print("invalid value encountered")
        print()


