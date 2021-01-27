'''
 This file define methods to load different type of data
'''
import io

from scipy import io as sio
from scipy.sparse import csr_matrix

from scipy.io import savemat
### To move later in a config file
DATA_INF = {
    "dblp":{
        "graph":{"filepath":"DATA/DBLP/Dblp.mat", "filetype":"matlab"},
        "communities":{"filepath":"DATA/DBLP/labels.txt", "filetype":"scc-t"},
        "directed":True
    },
    "wikipedia":{
        "graph":{"filepath":"DATA/wikipedia/wikipedia.mat", "filetype":"matlab"},
        "communities":{"filepath":"DATA/wikipedia/wikipedia.mat", "filetype":"matlab"},
        "directed":False
    },
    "blogCatalog":{
        "graph":{"filepath":"DATA/blogCatalog/edges.csv", "filetype":"scc"},
        "communities":{"filepath":"DATA/blogCatalog/group-edges.csv", "filetype":"scc"},
        "directed":False
    },
    "flickr":{
        "graph":{"filepath":"DATA/flickr/edges.csv", "filetype":"scc"},
        "communities":{"filepath":"DATA/flickr/group-edges.csv", "filetype":"scc"},
        "directed":False
    },
    "karate":{
        "graph":{"filepath":"DATA/karate/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/karate/communities.txt", "filetype":"txt"},
        "directed":False
    },   
    "books":{
        "graph":{"filepath":"DATA/books/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/books/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-1":{
        "graph":{"filepath":"DATA/LFR1/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR1/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-2":{
        "graph":{"filepath":"DATA/LFR2/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR2/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-3":{
        "graph":{"filepath":"DATA/LFR3/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR3/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-4":{
        "graph":{"filepath":"DATA/LFR4/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR4/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-5":{
        "graph":{"filepath":"DATA/LFR5/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR5/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-6":{
        "graph":{"filepath":"DATA/LFR6/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR6/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-7":{
        "graph":{"filepath":"DATA/LFR7/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR7/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-8":{
        "graph":{"filepath":"DATA/LFR8/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR8/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR-9":{
        "graph":{"filepath":"DATA/LFR9/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR9/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR1":{
        "graph":{"filepath":"DATA/LFR1/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR1/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR2":{
        "graph":{"filepath":"DATA/LFR2/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR2/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR3":{
        "graph":{"filepath":"DATA/LFR3/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR3/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR4":{
        "graph":{"filepath":"DATA/LFR4/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR4/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR5":{
        "graph":{"filepath":"DATA/LFR5/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR5/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR6":{
        "graph":{"filepath":"DATA/LFR6/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR6/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR7":{
        "graph":{"filepath":"DATA/LFR7/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR7/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR8":{
        "graph":{"filepath":"DATA/LFR8/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR8/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "LFR9":{
        "graph":{"filepath":"DATA/LFR9/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/LFR9/communities.txt", "filetype":"txt"},
        "directed":False
    }, 
    "polblogs":{
        "graph":{"filepath":"DATA/polblogs/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/polblogs/communities.txt", "filetype":"txt"},
        "directed":False
    },   
    "football":{
        "graph":{"filepath":"DATA/football/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/football/communities.txt", "filetype":"txt"},
        "directed":False
    },   
    "adjnoun":{
        "graph":{"filepath":"DATA/adjnoun/edges.txt", "filetype":"txt"},
        "communities":{"filepath":"DATA/adjnoun/communities.txt", "filetype":"txt"},
        "directed":False
    }
}


def directed_to_undirected(graph):
    """ 
    transform an directed graph into undirected one
  
    Parameters: 
    graph : a dictionary having for keys nodes and values list of neigbhor nodes
  
    Returns: 
    graph_undirected : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    graph_copy = {k:{node for node in v} for k, v in graph.items()}

    for k in range(len(graph)):
        v = graph[k]

        for node in v:
            if(k not in graph_copy[node]):
                graph_copy[node].add(k)
    return {k:list(v) for k,v in graph_copy.items()}

def reindex(graph, labels=None):
    """ 
    reindex the graph to get examples to 0-n_nodes
  
    Parameters: 
    graph : a dictionary having for keys nodes and values list of neigbhor nodes
    labels : a dictionary having for keys nodes and values list of labels

    Returns: 
    graph_undirected : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    index_map = {}
    
    graph_copy = {}
    for k,v in graph.items():
        v = graph[k]
        if(k not in index_map):
            index_map[k] = len(index_map)
            graph_copy[index_map[k]] = []

        for node in v:
            if(node not in index_map):
                index_map[node] = len(index_map)
                graph_copy[index_map[node]] = []
            graph_copy[index_map[k]].append(index_map[node])

    labels_copy = {}
    if(labels is not None):
        labels_copy = {index_map[k]:v for k, v in labels.items()}
    print(len(graph_copy), len(graph))
    print(max(list(graph_copy.values())))
    return graph_copy, labels_copy, index_map


def load_corpus(corpus_id, directed=None):
    """ 
    Loading graph
  
    Parameters: 
    corpus_id (str): The id of the corpus
    directed (bool) : optional argument by default graph are considered undirected

    Returns: 
    X, Y : a dictionary having for keys nodes and values list of neigbhor nodes 
           and a dictionary having for keys nodes and values list of communities
    """
    try:
        corpus_infos = DATA_INF[corpus_id]
        if(directed is None):
            directed = corpus_infos["directed"]
    except : 
        print("The required ",corpus_id," dataset is unknown")
        print("The indexed datasets are : ", list(DATA_INF.keys()))
        quit()
    X = LOAD_INF["graph"][corpus_infos["graph"]["filetype"]](corpus_infos["graph"]["filepath"])
    if(not directed):
        X = directed_to_undirected(X)
    Y = LOAD_INF["communities"][corpus_infos["communities"]["filetype"]](corpus_infos["communities"]["filepath"])
    # reindex from 0 to n_communities
    return X, Y

def save_matlab_edges(X, filepath, mat_key="network"):

    """ 
    saving edges and nodes of a graph, where graph is represented
    with the adjancy matrix from matlab format.
  
  
    Parameters: 
    mat_filepath (str): The filepath of the .mat file
    mat_key (str): The key corresponding to the adjancy matrix in the graph
  
    Returns: 
    X : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    row = []
    col = []
    data = []
    for start_node, end_nodes in X.items():
        for end_node in end_nodes:
            row.append(start_node)
            col.append(end_node)
            data.append(1.)
    matrix_to_save = csr_matrix((data, (row, col)))
    dic_to_save = {mat_key: matrix_to_save}
    savemat(filepath, dic_to_save)

def save_communities(Y, filepath):
    with  io.open(filepath, 'w') as save_file:
        print(Y)
        for node, communities in  Y.items():
            print(node)
            save_file.write(str(node))

            for community in communities:
                save_file.write("\t"+str(community))
            save_file.write('\n')


def read_matlab_graph(mat_filepath, mat_key="network"):
    """ 
    Loading edges and nodes of a graph, where graph is represented
    with the adjancy matrix from matlab format.
  
  
    Parameters: 
    mat_filepath (str): The filepath of the .mat file
    mat_key (str): The key corresponding to the adjancy matrix in the graph
  
    Returns: 
    X : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    
    data = sio.loadmat(mat_filepath)


    # getting edges and communities matrix
    edges = data[mat_key]
    # processing edges
    column, line = edges.nonzero()
    X = {}
    for i, (x, y) in enumerate(zip(column, line)):
        if(int(x) not in X):
            X[int(x)] = []
        if(int(y) not in X):
            X[int(y)] = []
        X[int(x)].append(int(y))
 
    return X

def read_matlab_communities(mat_filepath, mat_key="group"):
    """ 
    Loading communities from matlab format.
  
  
    Parameters: 
    mat_filepath (str): The filepath of the .mat file
    mat_key (str): The key corresponding to the communities matrix
  
    Returns: 
    Y : a dictionary having for keys nodes and values list of communities
    """
    data = sio.loadmat(mat_filepath)

    # getting community matrix
    communities = data[mat_key]

    column, line = communities.nonzero()
    Y = {}
    for i, (x, y) in enumerate(zip(column,line)):
        if(x not in Y):
            Y[x] = []
        Y[x].append(y) 
    return Y

def read_scc_graph(edges_filepath):
    """ 
    Loading edges and nodes of a graph, where graph is in social 
    computing corpus format
  
  
    Parameters: 
    edges_filepath (str): The filepath of edges
  
    Returns: 
    X : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    X = {}
    with io.open(edges_filepath, "r") as edges_file:
        for line in edges_file:
            lsp = line.split(",")
            if(int(lsp[0])-1 not in X):
                X[int(lsp[0])-1] = []
            if(int(lsp[1])-1 not in X):
                X[int(lsp[1])-1] = []
            X[int(lsp[0])-1].append(int(lsp[1])-1)

    return X

def read_scc_communities(communities_filepath):
    """ 
    Loading communities, where communities file is in social 
    computing corpus format
  
  
    Parameters: 
    communities_filepath (str): The filepath of communities

    Returns: 
    Y : a dictionary having for keys nodes and values list of communities
    """
    Y = {}
    with io.open(communities_filepath, "r") as label_file:
        for line in label_file:
            lsp = line.split(",")
            if(int(lsp[0])-1 not in Y):
                Y[int(lsp[0])-1] = []
            Y[int(lsp[0])-1].append(int(lsp[1])-1)
    return Y

def read_scc_t_communities(communities_filepath):
    """ 
    Loading communities, where communities file is in social 
    computing corpus format
  
  
    Parameters: 
    communities_filepath (str): The filepath of communities

    Returns: 
    Y : a dictionary having for keys nodes and values list of communities
    """
    Y = {}
    with io.open(communities_filepath, "r") as label_file:
        for line in label_file:
            lsp = line.split()
            if(int(lsp[0])-1 not in Y):
                Y[int(lsp[0])-1] = []
            Y[int(lsp[0])-1].append(int(lsp[1])-1)
    return Y

def read_txt_graph(graph_filepath):
    """ 
    Loading edges and nodes of a graph, where graph is represented
    with the txt matrix format
  
  
    Parameters: 
    graph_filepath (str): The filepath of the graph file

  
    Returns: 
    X : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    X = {}
    with io.open(graph_filepath, "r") as edges_file:
        for i, line in enumerate(edges_file):
            lsp = line.split()
            X[i] = [k for k, value in enumerate(lsp) if(int(value) == 1)]
    return X

def read_txt_communities(communities_filepath):
    """ 
    Loading communities from matlab format.
  
  
    Parameters: 
    communities_filepath (str): The filepath of the .mat file

  
    Returns: 
    Y : a dictionary having for keys nodes and values list of communities
    """
    Y = {}
    with io.open(communities_filepath, "r") as label_file:
        for i, line in enumerate(label_file):
            
            Y[i] = []
            Y[i].append(int(line)-1)
    return Y


def read_xml_graph(graph_filepath):
    """ 
    Loading edges and nodes of a graph, where graph is represented
    with the xml format
  
  
    Parameters: 
    graph_filepath (str): The filepath of the graph file

  
    Returns: 
    X : a dictionary having for keys nodes and values list of neigbhor nodes
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(graph_filepath)
    root = tree.getroot()
    graph = root.find("graph")
    nodes = root.find("graph").find("nodes").findall("node")
    edges = root.find("graph").find("edges").findall("edge")
    X = {}
    for edge in edges:
        n_from, n_to = int(edge.attrib["vertex1"]), int(edge.attrib["vertex2"])
        if(n_from not in X):
            X[n_from] = []
        if(n_to not in X):
            X[n_to] = []
        X[n_from].append(n_to)
    return X



LOAD_INF = {
    "graph":{
        "matlab": read_matlab_graph, 
        "scc": read_scc_graph, 
        "txt": read_txt_graph
        },
    "communities":{
        "matlab": read_matlab_communities,
        "scc": read_scc_communities,
        "scc-t": read_scc_t_communities, 
        "txt": read_txt_communities
    }
}
