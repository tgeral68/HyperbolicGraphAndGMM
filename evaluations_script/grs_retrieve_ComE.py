'''
This script read the json and print performances obtained regrouping by dimenssion and dataset
'''
import argparse
import torch
from rcome.data_tools import logger
import os
import sys
parser = argparse.ArgumentParser(description='Load embeddings and perform kmeans on it')


parser.add_argument('--folder', dest="folder", type=str, default="",
                    help="The folder where to find experiments") 
parser.add_argument('--folder-log-substring', dest="fls", type=str, default=".*",
                    help="The format of the log")
parser.add_argument('--select-index', dest="select_index", type=str, default="unsupervised_eval_conductance_N",
                    help="select performance according to the metric")          
parser.add_argument('--verbose', dest="verbose", default=False, action="store_true",
                    help="The format of the log")        
args = parser.parse_args()


list_of_directory = os.listdir(args.folder)

experiments_res = {}
for directory in list_of_directory:
    if(args.fls not in directory and args.fls != ".*"):
        # list_of_directory.remove(list_of_directory)
        pass
    else:
        try:

            logger_object = logger.JSONLogger(os.path.join(args.folder,directory,"log.json"), mod="continue")
            dct_key = logger_object["dataset"]+"-"+str(logger_object["size"])+"D"
            experiment_identifier = logger_object['id']
            if(dct_key not in experiments_res):
                experiments_res[dct_key] = {}
            if(args.verbose):
                print("Folder \""+directory+"\" embedding size->"+str(logger_object["size"]))

            # ll = logger_object["supervised_eval"]["linear_logit"]
            # if("supervised_eval_linear_logit_p1" not in experiments_res[dct_key]):
            #     experiments_res[dct_key]["supervised_eval_linear_logit_p1"] = {}
            # if(experiment_identifier not in experiments_res[dct_key]["supervised_eval_linear_logit_p1"]):
            #     experiments_res[dct_key]["supervised_eval_linear_logit_p1"][experiment_identifier] = ll["P1"]

            # if("supervised_eval_linear_logit_p3" not in experiments_res[dct_key]):
            #     experiments_res[dct_key]["supervised_eval_linear_logit_p3"] = {}
            # if(experiment_identifier not in experiments_res[dct_key]["supervised_eval_linear_logit_p3"]):
            #     experiments_res[dct_key]["supervised_eval_linear_logit_p3"][experiment_identifier] = ll["P3"]

            # if("supervised_eval_linear_logit_p5" not in experiments_res[dct_key]):
            #     experiments_res[dct_key]["supervised_eval_linear_logit_p5"] = {}
            # if(experiment_identifier not in experiments_res[dct_key]["supervised_eval_linear_logit_p5"]):
            #     experiments_res[dct_key]["supervised_eval_linear_logit_p5"][experiment_identifier] = ll["P5"]

            ll = logger_object["unsupervised_eval"]["accuracy"]
            if("unsupervised_eval_accuracy" not in experiments_res[dct_key]):
                experiments_res[dct_key]["unsupervised_eval_accuracy"] = {}
            if(experiment_identifier not in experiments_res[dct_key]["unsupervised_eval_accuracy"]):
                experiments_res[dct_key]["unsupervised_eval_accuracy"][experiment_identifier] = [ll]


            ll = logger_object["unsupervised_eval"]["conductance"]
            if("unsupervised_eval_conductance_N" not in experiments_res[dct_key]):
                experiments_res[dct_key]["unsupervised_eval_conductance_N"] = {}
            if(experiment_identifier not in experiments_res[dct_key]["unsupervised_eval_conductance_N"]):
                experiments_res[dct_key]["unsupervised_eval_conductance_N"][experiment_identifier] = \
                    [1-(sum(ll, 0)/len(ll))]

            ll = logger_object["unsupervised_eval"]["nmi"]
            if("unsupervised_eval_nmi" not in experiments_res[dct_key]):
                experiments_res[dct_key]["unsupervised_eval_nmi"] = {}
            if(experiment_identifier not in experiments_res[dct_key]["unsupervised_eval_nmi"]):
                experiments_res[dct_key]["unsupervised_eval_nmi"][experiment_identifier] = \
                    [ll]

        except Exception as e:
            print("An error occured reading "+directory+" log, this folder will be ignored")
            print(e)

# printing json files

for dct_key, dct_val in experiments_res.items():
    try:
        print("Results for dataset "+dct_key)
        if(args.select_index != ""):
            best_xp_name = ""
            best_xp_res = 0.
            # reading the metric
            metric_dct = dct_val[args.select_index]
            for xp_name, xp_res in metric_dct.items():
                res = sum(xp_res, 0)/(len(xp_res))
                if(res > best_xp_res):
                    best_xp_res = res
                    best_xp_name = xp_name
            if("N" in args.select_index):
                best_xp_res = 1-best_xp_res
            print("\t\t Best Performances obtain with xp ", best_xp_name, " according to ", args.select_index)
            for metric, dct_xp in dct_val.items():  
                # print(metric, dct_xp) 
                # print()                     
                xp_res = dct_xp[best_xp_name]
                res = sum(xp_res, 0)/(len(xp_res))
                if("N" in metric):
                    res = 1-res
                print("\t\t\t", metric, " scoring at ", res)
        else:
            for metric, dct_xp in dct_val.items():

                print("\tMetric -> ", metric)

                best_xp_name = ""
                best_xp_res = 0.
                for xp_name, xp_res in dct_xp.items():

                    res = sum(xp_res, 0)/(len(xp_res))
                    if(res > best_xp_res):
                        best_xp_res = res
                        best_xp_name = xp_name
                if("N" in metric):
                    best_xp_res = 1-best_xp_res
                print("\t\t Best Performances obtain with xp ", best_xp_name, " scoring at ", best_xp_res)

    except Exception:
        print("invalid value encountered")
        print()


