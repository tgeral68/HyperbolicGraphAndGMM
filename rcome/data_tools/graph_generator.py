import io
import torch
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_pylab import draw
from networkx.generators.community import LFR_benchmark_graph


def lancichinetti_fortunato_radicchi_graph(n_nodes, mu, gamma, beta):
    graph = LFR_benchmark_graph(n_nodes, gamma, beta, mu, min_community=50, max_community=80, average_degree=6, max_iters=2000)

    graph.remove_nodes_from(nx.isolates(graph))
    communities = {frozenset(graph.nodes[v]["community"]) for v in graph} 
    communities_tensor = torch.zeros(len(graph))
    print("Number of communities ", len(communities))
    for community_index, community in enumerate(communities):
        for node in community:
            communities_tensor[node] = community_index
    graph_set = {}
    for i in graph:
        graph_set[i] = list(graph[i])

    return graph, graph_set, communities_tensor

def save_community_graph(graph, communities, filepath_txt, filepath_com):
    with io.open(filepath_txt, 'w') as graph_file:
        for i in range(len(graph)):
            for j in range(len(graph)):
                if(j in graph[i]):
                    graph_file.write('1')
                else :
                    graph_file.write('0')
                if(j != len(graph) -1):
                    graph_file.write('\t')
            graph_file.write('\n') 
    with io.open(filepath_com, 'w') as graph_file:
        for com in communities:
            graph_file.write(str(int(com.item())+1)+'\n')

def _test_LFR():
    graph, graph_set, communities = lancichinetti_fortunato_radicchi_graph(800, 0.08, 3, 1.5)
    save_community_graph(graph_set, communities, "DATA/LFR/edges.txt", "DATA/LFR/communities.txt")
    print(communities)
    draw(graph)
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    # execute all the test
    _test_LFR()
