import nltk

def animal_recursive(synset, parent_set=[], depth=1, max_depth=5):
    if(depth > max_depth or len(synset.hyponyms()) == 0):
        return [parent_set]
    synset_list = []
    for s in synset.hyponyms():
        x = animal_recursive(s, parent_set=(parent_set+[s.name()]),
                           depth=depth+1, max_depth=max_depth)
        synset_list += x
    return synset_list

def animals(max_depth=8, root='animal.n.01'):
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    root = wordnet.synset(root)
    animals_dataset = animal_recursive(root, [root.name()], max_depth=max_depth)
    dictionary = {}
    tuple_neigbhor = []
    X = set()
    for document in animals_dataset:
        ll = []
        for l in document:
            if l not in dictionary:
                dictionary[l] = len(dictionary)+1
            for l2 in document:
                if l2 not in dictionary:
                    dictionary[l2] = len(dictionary)+1   
                if l != l2 :          
                    ll.append((dictionary[l], dictionary[l2]))
        for i in range(len(document)-1):
            tuple_neigbhor.append((dictionary[document[i]], dictionary[document[i+1]]))
        X = X.union(set(ll))

    return list(X), dictionary, animals_dataset, tuple_neigbhor



if __name__ == "__main__": 
    test_animal()