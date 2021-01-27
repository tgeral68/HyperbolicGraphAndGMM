from rcome.data_tools import data_loader, corpora

X, Y = data_loader.load_corpus("LFR1", directed=False)

# Size of the dataset precompute * path_len
dataset = corpora.RandomContextSizeFlat(
        X, Y, precompute=2, path_len=10, context_size=3)
# Tuples (node, context) for the context of path 1 position 1
tuple_node_context = dataset[1]

print('Example of a (node, context) tuple:', tuple_node_context)