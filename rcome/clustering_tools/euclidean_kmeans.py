import math
import cmath
import torch
import numpy as np
import tqdm
import sklearn.cluster as skc
import sklearn.mixture as skm
import random
from rcome.function_tools import euclidean_function as ef

class sklearnKMeans(object):
    def __init__(self, n_clusters, min_cluster_size=2, verbose=False, init_method="random"):
        self.kmeans = skc.KMeans(n_clusters=n_clusters, max_iter=3000)

    def fit(self, X, Y=None):
        self.kmeans.fit(X.numpy())

    def predict(self, X):
        return torch.Tensor(self.kmeans.predict(X.numpy()))

    def probs(self, X):
        predicted = self.predict(X).squeeze().tolist()
        res = torch.zeros(len(X), self._n_c)
        for i, l in enumerate(predicted):
            res[i][l] = 1
        return res


class KMeans(object):
    def __init__(self, n_clusters, min_cluster_size=2, verbose=False, init_method="random"):
        self._n_c = n_clusters
        self._distance = ef.distance
        self.centroids = None
        self._mec = min_cluster_size
        self._init_method = init_method

    def _maximisation(self, x, indexes):
        centroids = x.new(self._n_c, x.size(-1))
        for i in range(self._n_c):
            lx = x[indexes == i]
            if(lx.shape[0] <= self._mec):
                lx = x[random.randint(0,len(x)-1)].unsqueeze(0)
            centroids[i] = lx.mean(0)
        return centroids
    
    def _expectation(self, centroids, x):
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)
        value, indexes = dst.min(-1)
        return indexes

    def _init_random(self, X):
        self.centroids_index = (torch.rand(self._n_c, device=X.device) * len(X)).long()
        self.centroids = X[self.centroids_index]

    def __init_kmeansPP(self, X):
        distribution = torch.ones(len(X))/len(X)
        frequency = pytorch_categorical.Categorical(distribution)
        centroids_index = []
        N, D = X.shape
        while(len(centroids_index)!=self._n_c):

            f = frequency.sample(sample_shape=(1,1)).item()
            if(f not in centroids_index):
                centroids_index.append(f)
                centroids = X[centroids_index]
                x = X.unsqueeze(1).expand(N, len(centroids_index), D)
                dst = self._distance(centroids, x)
                value, indexes = dst.min(-1)
                vs = value**2
                distribution = vs/(vs.sum())
                frequency = pytorch_categorical.Categorical(distribution)
        self.centroids_index = torch.tensor(centroids_index, device=X.device).long()
        self.centroids = X[self.centroids_index]

    def fit(self, X, Y=None, max_iter=500):
        if(Y is None):
            with torch.no_grad():
                if(self._mec < 0):
                    self._mec = len(X)/(self._n_c**2)
                if(self.centroids is None):
                    if(self._init_method == "kmeans++"):
                        self.__init_kmeansPP(X)
                    else:
                        self._init_random(X)
                for iteration in range(max_iter):
                    if(iteration >= 1):
                        old_indexes = self.indexes
                    self.indexes = self._expectation(self.centroids, X)
                    self.centroids = self._maximisation(X, self.indexes)
                    if(iteration >= 1):   
                        if((old_indexes == self.indexes).float().mean() == 1):
                            # print(" Iteration end : ", iteration)
                            self.cluster_centers_  =  self.centroids
                            return self.centroids
                self.cluster_centers_  =  self.centroids
                return self.centroids
        else:
            # print("lalalaalalalal")
            # print(Y.size())
            self.indexes = Y.max(-1)[1]
            self.centroids = self._maximisation(X, self.indexes)
            self.cluster_centers_  =  self.centroids
            return self.centroids
    def predict(self, X):
        return self._expectation(self.centroids, X)

    def probs(self, X):
        predicted = self._expectation(self.centroids, X).squeeze().tolist()
        res = torch.zeros(len(X), self._n_c)
        for i, l in enumerate(predicted):
            res[i][l] = 1
        return res

    def getStd(self, x):
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = self.centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)**2
        value, indexes = dst.min(-1)
        stds = []
        for i in range(self._n_c):
            stds.append(value[indexes==i].sum())
        stds = torch.Tensor(stds)
        return stds

