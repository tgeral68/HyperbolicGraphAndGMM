import io
import random
import tqdm
import time

# torch lib import 
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For loading matlab graph files
from scipy import io as sio


class DatasetTuple(Dataset):
    def __init__(self, X, verbose=1):
        self.X = X

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):

        return torch.Tensor([self.X[index][0], self.X[index][1]])

class RandomWalkCorpus(Dataset):
    def __init__(self, X, Y, path=True):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y
        self.k = 0
        self.path = path
        self.p_c = 1

    def set_walk(self, maximum_walk, continue_probability):
        self.k = maximum_walk
        self.p_c = continue_probability

    def set_path(self, path_val):
        self.path = path_val

    def light_copy(self):
        rwc_copy =  RandomWalkCorpus(self.X, self.Y, path=self.path)
        rwc_copy.k = self.k
        rwc_copy.p_c = self.p_c

        return rwc_copy
    def getFrequency(self):
        return torch.Tensor([[k, len(v)] for k, v in self.X.items()])
    def _walk(self, index):
        path = []
        c_index = index 
        path.append(c_index)
        for i in range(self.k):
            
            if(random.random()>self.p_c):
                break

            c_index = self.X[c_index][random.randint(0,len(self.X[c_index])-1)]
            path.append(c_index)
        return path if(self.path) else [c_index] 

    def __getitem__(self, index):
        return torch.LongTensor([self._walk(index)]), torch.LongTensor(self.Y[index])

    def __len__(self):
        return len(self.X)

class NeigbhorFlatCorpus(Dataset):
    def __init__(self, X, Y):
        # the sparse torch dictionary
        self.X = X
        self.Y = Y

        self.data = []
        for ns, nln in X.items():
            for nl in nln:
                self.data.append([ns, nl])
        self.data = torch.LongTensor(self.data)

    def cuda(self):
        self.data = self.data.cuda()

    def cpu(self):
        self.data = self.data.cpu()

    def __getitem__(self, index):
        if(type(index) == int):
            a, b = self.data[index][0], self.data[index][1]
        else:
            a,b  = self.data[index][:,0], self.data[index][:,1]
        return a, b
    def __len__(self):
        return len(self.data)


class FlatContextCorpus(Dataset):
    def __init__(self, dataset, context_size=5, precompute=1):
        self._dataset = dataset
        self.c_s = context_size
        self.precompute = precompute
        if(precompute < 1):
            print("Precompute is mandatory value "+str(precompute)+ " must be a positive integer instead")
            precompute = 1
        self.context = torch.LongTensor(self._precompute()).unsqueeze(-1)
        self.n_sample = 5

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        context = [[] for i in range(len(self._dataset))]
        for p in range(precompute):
            for i in tqdm.trange(len(self._dataset)):
                # get the random walk
                path = self._dataset[i][0].squeeze()
                for k in range(len(path)):
                    for j in range(max(0, k - self.c_s), min(len(path), k + self.c_s)):
                        if(k!=j):
                            context[path[k].item()].append(path[j].item())
        flat_context = []
        for i, v  in enumerate(context):
            for item in v:
                flat_context.append([i, item])
        
        return flat_context

    def cuda(self):
        print("max index ", self.context.max())
        self.context = self.context.cuda()

    def cpu(self):
        self.context = self.context.cpu()

    def __getitem__(self, index):
        if(type(index) == int):
            a, b = self.context[index][0], self.context[index][1]
        else:
            a,b  = self.context[index][:,0], self.context[index][:,1]
        return a, b
    def __len__(self):
        return len(self.context)


class RandomContextSize(RandomWalkCorpus):
    def __init__(self, X, Y, path=True, precompute=1, path_len=10, context_size=5):
        super(RandomContextSize, self).__init__(X, Y, path=path)
        self.precompute = precompute
        self.k = path_len
        self.p_c = 1
        self.c_s = context_size
        self.paths = self._precompute()

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        paths = []
        for index in tqdm.trange(len(self.X)):
            for i in range(precompute):
                    paths.append(torch.LongTensor(self._walk(index)).unsqueeze(0))
        self.precompute = precompute
        return torch.cat(paths,0)


    def cuda(self):
        self.paths = self.paths.cuda()

    def cpu(self):
        self.paths = self.paths.cpu()

    @staticmethod
    def _context_calc(k, path, max_context):
        context_size = random.randint(1, max_context -1)
        v = torch.cat((path[max(0, k - context_size):k],path[k+1: min(len(path), k + context_size)]) ,0)
        v =  v.unsqueeze(-1).expand(v.size(0), 2).clone()
        v[:,0] = path[k]
        return v

    def __getitem__(self, index):
        path = self.paths[index]

        res = [RandomContextSize._context_calc(i, path, self.c_s) for i in range(len(path))]

        return torch.cat(res, 0)

    def __len__(self):
        return len(self.paths)

class RandomContextSizeFlat(RandomWalkCorpus):
    def __init__(self, X, Y, path=True, precompute=1, path_len=10, context_size=5):
        super(RandomContextSizeFlat, self).__init__(X, Y, path=path)
        self.precompute = precompute
        self.k = path_len
        self.p_c = 1
        self.c_s = context_size
        self.paths = self._precompute()

    def _precompute(self):
        precompute = self.precompute
        self.precompute = -1
        paths = []
        for index in tqdm.trange(len(self.X)):
            for i in range(precompute):
                    paths.append(torch.LongTensor(self._walk(index)).unsqueeze(0))
        self.precompute = precompute
        return torch.cat(paths,0)


    def cuda(self):
        self.paths = self.paths.cuda()

    def cpu(self):
        self.paths = self.paths.cpu()

    @staticmethod
    def _context_calc(k, path, max_context):
        context_size = random.randint(1, max_context -1)
        v = torch.cat((path[max(0, k - context_size):k],path[k+1: min(len(path), k + context_size)]) ,0)
        v =  v.unsqueeze(-1).expand(v.size(0), 2).clone()
        v[:,0] = path[k]
        return v

    def __getitem__(self, index):

        path = self.paths[index//self.k]
        # print(path)
        res = RandomContextSize._context_calc(index%self.k, path,self.c_s) 
        # print("here", res)
        return res

    def __len__(self):
        return len(self.paths) * self.k

