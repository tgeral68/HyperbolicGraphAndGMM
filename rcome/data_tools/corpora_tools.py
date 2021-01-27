from torch.utils.data import Dataset
import torch
import tqdm
import random

########################## DATASET OBJECT ############################
class ZipDataset(Dataset):
    def __init__(self, *args):
        self.datasets = args

        # checking all dataset have the same length
        self.len_dataset = None
        for dataset in self.datasets:
            if(self.len_dataset is None):
                self.len_dataset = len(dataset)
            else:
                assert(self.len_dataset == len(dataset))
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        return sum([dataset[index] for dataset in self.datasets] ,())

class IndexableDataset(Dataset):
    def __init__(self, L):
        self.indexable_structure = L
    
    def __len__(self):
        return len(self.indexable_structure)
    
    def __getitem__(self, index):
        return (self.indexable_structure[index],)
class SelectFromPermutation(Dataset):
    def __init__(self, dataset, permutation):
        self.dataset = dataset
        self.permutation = permutation
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        permutation_index = self.permutation[index]
        if(type(permutation_index) == tuple):
            permutation_index = permutation_index[0]
        if(type(permutation_index) != int):
            if(type(permutation_index) == torch.Tensor):
                permutation_index = int(permutation_index.item())
            elif(type(permutation_index) == float):
                permutation_index = int(permutation_index)
        res = self.dataset[permutation_index]
        if(type(res) == tuple):
            return res
        else:
            return (res,)
class SelectFromIndexDataset(Dataset):
    def __init__(self, dataset, element_index=0):
        self.indexable_structure = dataset
        self.element_index = element_index
    def __len__(self):
        return len(self.indexable_structure)
    
    def __getitem__(self, index):
        return (self.indexable_structure[index][self.element_index],)    


class SamplerDataset(Dataset):
    def __init__(self, dataset, n_sample=5, policy=None):
        self.dataset = dataset
        self.n_sample = n_sample
        self.policy = policy


    def __getitem__(self, index):
        select = self.dataset[index][0]
        if(len(select)!=0):
            if(self.policy is not None):
                probs = self.policy[select-1]/(self.policy[select-1].sum())
                distrib = torch.distributions.Categorical(probs=probs)
                rus = tuple([torch.LongTensor([select[distrib.sample()]]) for i in range(self.n_sample)])
            else:
                rus = tuple([torch.LongTensor([select[random.randint(0, len(select)-1)]]) for i in range(self.n_sample)])
        else:
            rus = tuple([torch.LongTensor([0]) for i in range(self.n_sample)])
        return rus

    def __len__(self):
        return len(self.dataset)

class RepeaterDataset(Dataset):
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size
        self.real_len = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index%self.real_len]

    def __len__(self):
        return self.size 
############################ FACTORIES #############################

def repeat_dataset(dataset, size):
    return RepeaterDataset(dataset, size)

def zip_datasets(*args):
    return ZipDataset(*args)

def from_indexable(L):
    return IndexableDataset(L)

def select_from_index(dataset, element_index=0):
    return SelectFromIndexDataset(dataset, element_index=element_index)

def knn(X, Y, distance, k=10, n_sample=10, alpha=3000):
    return KNNDataset(X,Y, distance, k=k, n_sample=n_sample, alpha=alpha)

def sampler(dataset, n_sample=1, policy=None):
    return SamplerDataset(dataset, n_sample=n_sample, policy=policy)

def permutation_index(dataset, permutation):
    return SelectFromPermutation(dataset, permutation)