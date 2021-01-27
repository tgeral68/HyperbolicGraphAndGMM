import argparse
import io
import json

parser = argparse.ArgumentParser(description='Grid search generator from json file for shell and mongo')

parser.add_argument('--file', metavar='f', dest='parameter_file', type=str,
                    help='JSON File containing the parameters')
parser.add_argument('--cmd', metavar='c', dest='cmd', type=str,
                     help='The command to launch')
parser.add_argument('--out', metavar='o', dest='out', type=str,
                     help='The output file where storing gs shell ')
parser.add_argument('--type', metavar='t', dest='gs_type', type=str, default="shell",
                    help='The type of the grid search [shell or mongo]')
parser.add_argument('--database-name', metavar='d', dest="database", type=str, default="",
                    help="if use mongo the database where will be registered waiting xps")
parser.add_argument('--server-name', metavar='s', dest="server_name", type=str, default="",
                    help="if use mongo the name of the mongo server")
parser.add_argument('--collection-name', dest="collection_name", type=str, default="queue",
                    help="if use mongo, name of the collection where experiments queue is set")                 
args = parser.parse_args()



def generate_shell_grid_search(parameters_file, cmd, out=""):
    json_data=open(parameters_file).read()
    hp = json.loads(json_data)
    keys = list(hp.keys())
    run_list = rec_parameters(hp, keys, 0)

    with io.open(out,'w') as out_file:
        for i, run in enumerate(run_list):
            out_file.write((cmd+' '+run+' --id '+hp["dataset"][0]+'-grid-'+str(i)+'\n'))

def generate_mongo_grid_search(parameters_file, cmd, server_name, database_name, collection_name):
    import pymongo
    from pymongo import MongoClient

    mc = MongoClient(host=server_name) 
    database =  mc[database_name]
    mongo_collection = database[collection_name]
    json_data=open(parameters_file).read()
    hp = json.loads(json_data)
    keys = list(hp.keys())
    run_list = rec_parameters_list(hp, keys, 0)
    print(run_list)
    for run in run_list:
        mongo_collection.insert({"status":"waiting", "exec":cmd, "parameters":run})



def rec_parameters(parameters, keys, index_keys):
    if(index_keys>=len(parameters)):
        return ['']
    c_list = []
    for ss in rec_parameters(parameters, keys, index_keys+1):
        for v in parameters[keys[index_keys]]:
            if(str(v) == ""):
                c_list.append(" --"+str(keys[index_keys])+" "+ss)
            else:
                c_list.append(" --"+str(keys[index_keys])+" "+str(v)+ss)
    return c_list

def rec_parameters_list(parameters, keys, index_keys):
    if(index_keys>=len(parameters)):
        return [dict()]
    c_list = []
    for ss in rec_parameters_list(parameters, keys, index_keys+1):
        for v in parameters[keys[index_keys]]:
            n_dct = {str(keys[index_keys]):v}
            n_dct.update(ss)
            c_list.append(n_dct)
    return c_list

if(args.gs_type == "shell"):
    generate_shell_grid_search(args.parameter_file, args.cmd, args.out)
elif(args.gs_type == "mongo"):
    generate_mongo_grid_search(args.parameter_file, args.cmd, args.server_name, args.database, args.collection_name)