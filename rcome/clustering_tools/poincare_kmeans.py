import math
import cmath
import torch
import numpy as np
import tqdm
import random
import time

from rcome.function_tools import poincare_alg as pa
from rcome.function_tools import poincare_function as pf
from rcome.function_tools import pytorch_categorical


## in the following we define methods to initialize k-means centroids

def _init_random(X, n_clusters, distance):
    centroids_index = (torch.rand(n_clusters, device=X.device) * len(X)).long()
    centroids_value = X[centroids_index]
    centroids_value = centroids_value.to(X.device)

    return centroids_index, centroids_value

def _init_kmeansPP(X, n_clusters, distance):
    ## initial distribution

    distribution_values = torch.ones(len(X))/len(X)
    distribution = pytorch_categorical.Categorical(distribution_values)

    ## empty centroids
    centroids_index = []
    N, D = X.shape

    ## fill centroids until reaching number of clusters

    while len(centroids_index) != n_clusters:

        ## sample over the current distribution (can also
        ## use argmax in a non stochastic process)
        sampled_point = distribution.sample(sample_shape=(1,1)).item()

        ## if the sampled point is not an already selected centroid
        if(sampled_point not in centroids_index):
        
            centroids_index.append(sampled_point)
            centroids_value = X[centroids_index]

            ## compute the distance of each point to each centroids
            x = X.unsqueeze(1).expand(N, len(centroids_index), D)
            dst = distance(centroids_value, x)

            ## selecting for each point only the closest centroids distance
            ## and store the squared distance associated 
            value, indexes = dst.min(-1)
            squared_distance = value**2

            ## normalise the distance according to other distance
            ## in order to get the distribution value
            distribution_value = squared_distance/(squared_distance.sum())

            ## build the new distribution over examples
            distribution = pytorch_categorical.Categorical(distribution_value)

    # transform data to pytorch tensor
    centroids_index = torch.tensor(centroids_index, device=X.device).long()
    centroids_value = X[centroids_index]

    # return centroids index and value
    return centroids_index, centroids_value


class PoincareKMeansNInit(object):
    def __init__(self, n_clusters, min_cluster_size=5, verbose=False, init_method="kmeans++", n_init=20):
        self.verbose = verbose
        self.KMeans = [PoincareKMeans(n_clusters, min_cluster_size, verbose, init_method) for i in range(n_init)]
    
    def fit(self, X, Y=None, max_iter=10):
        pb = range(len(self.KMeans))
        stds = torch.zeros(len(self.KMeans))

        for i, kmeans in zip(pb,self.KMeans):
            kmeans.fit(X, Y, max_iter)
            stds[i] = kmeans.getStd(X).mean()
        self.min_std_val, self.min_std_index = stds.min(-1)
        self.min_std_val, self.min_std_index = self.min_std_val.item(), self.min_std_index.item()
        self.kmean = self.KMeans[self.min_std_index]
        self.centroids = self.kmean.centroids
        self.cluster_centers_  =  self.centroids 
        self.KMeans = None

    def predict(self, X):
        return self.kmean._expectation(self.centroids, X)

    def getStd(self, X):
        return self.kmean.getStd(X)

class PoincareKMeans(object):
    def __init__(self, n_clusters, min_cluster_size=2, verbose=False, init_method="kmeans++"):
        self._init_methods_set = {'random':_init_random, 'kmeans++': _init_kmeansPP}

        self._n_c = n_clusters
        self._distance = pf.distance
        self.centroids = None
        self._mec = min_cluster_size
        self._init_method = self._init_methods_set[init_method]
        self.verbose = verbose

    def _fast_maximisation(self, x , indexes, batch_size=10):
        N, D = x.size(0), x.size(-1)
        start_time = time.time()
        centroids = x.new(self._n_c, x.size(-1))
        barycenter_time = 0
        mask_matrix = torch.zeros(x.size(0), self._n_c).to(x.device)
        for i in range(self._n_c):
            lx = x[indexes == i]
            if(lx.shape[0] <= self._mec):
                lx = x[random.randint(0,len(x)-1)].unsqueeze(0)
                indexes[random.randint(0,len(x)-1)] = i
            mask_matrix[indexes == i, i] = 1

        nb_batch = (self._n_c // batch_size) + (1 if((self._n_c % batch_size)!=0) else 0)

        for i in range(nb_batch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, self._n_c)
            weight = mask_matrix[:, start_index:end_index]
            barycenter_start_time = time.time()
            centroids[start_index:end_index] = pa.barycenter(x.unsqueeze(1).expand(N, end_index-start_index,
                                                                                    D), wik=weight,
                                                                                    normed=True,
                                                                                    lr=5e-3, tau=1e-3,
                                                                                    verbose=self.verbose)
            barycenter_end_time = time.time()
            barycenter_time += (barycenter_end_time - barycenter_start_time)
        end_time = time.time()
        if(self.verbose) :
            print("Fast Maximisation Time ", end_time-start_time)
            print("Cumulate barycenter Time ", barycenter_time)
        return centroids

    def _maximisation(self, x, indexes):
        start_time = time.time()
        centroids = x.new(self._n_c, x.size(-1))
        barycenter_time = 0
        for i in range(self._n_c):
            lx = x[indexes == i]
            if(lx.shape[0] <= self._mec):
                lx = x[random.randint(0,len(x)-1)].unsqueeze(0)
            barycenter_start_time = time.time()
            centroids[i] = pa.barycenter(lx, normed=True)
            barycenter_end_time = time.time()
            barycenter_time += (barycenter_end_time - barycenter_start_time)
        end_time = time.time()
        if(self.verbose) :
            print("Maximisation Time ", end_time-start_time)
            print("Cumulate barycenter Time ", barycenter_time)
        return centroids
    
    def _expectation(self, centroids, x):
        start_time = time.time()
        N, K, D = x.shape[0], self.centroids.shape[0], x.shape[1]
        centroids = centroids.unsqueeze(0).expand(N, K, D)
        x = x.unsqueeze(1).expand(N, K, D)
        dst = self._distance(centroids, x)
        value, indexes = dst.min(-1)
        end_time = time.time()
        if(self.verbose) :
            print("Expectation Time ", end_time-start_time)
        return indexes

    def fit(self, X, Y=None, max_iter=100):
        lt = []
        ft = 0
        if(Y is None):
            with torch.no_grad():
                if(self._mec < 0):
                    self._mec = len(X)/(self._n_c**2)
                if(self.centroids is None):
                    centroids_index, centroids_value = self._init_method(X, self._n_c, self._distance)
                    self.centroids = centroids_value
                    self.indexes = centroids_index
                for iteration in range(max_iter):
                    if(iteration >= 1):
                        old_indexes = self.indexes
                    start_time = time.time()
                    self.indexes = self._expectation(self.centroids, X)
                    self.centroids = self._fast_maximisation(X, self.indexes)
                    end_time = time.time()


                    if(iteration >= 1):   
                        if((old_indexes == self.indexes).float().mean() == 1):
                            self.cluster_centers_  =  self.centroids
                            return self.centroids
                self.cluster_centers_  =  self.centroids
                return self.centroids
        else:
            self.indexes = Y.max(-1)[1]
            self.centroids = self._maximisation(X, self.indexes)
            self.cluster_centers_  =  self.centroids

            return self.centroids

    def predict(self, X):
        return self._expectation(self.centroids, X)

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
    def probs(self, X):
        predicted = self._expectation(self.centroids, X).squeeze().tolist()
        res = torch.zeros(len(X), self._n_c)
        for i, l in enumerate(predicted):
            res[i][l] = 1
        return res


def _test_init_kmeans_pp():
    import matplotlib.pyplot as plt
    X = torch.Tensor([[0.5,0.5],[0.4,0.5],[0.5,0.4],
                      [-0.5,-0.5],[-0.4,-0.5],[-0.5,-0.4],
                      [-0.5,0.5],[-0.4,0.5],[-0.5,0.4]])
    distance = pf.distance
    n_clusters = 3
    centroids_index, centroids_value = _init_kmeansPP(X, n_clusters, distance)
    plt.scatter(X[:,0].numpy(), X[:,1].numpy(), label="point")
    plt.scatter(centroids_value[:,0].numpy(), centroids_value[:,1].numpy(), label="initial centroids")
    plt.title("Test $K$-Means++ initialisation")
    plt.legend()
    plt.show()

def _test_init_random():
    import matplotlib.pyplot as plt
    X = torch.Tensor([[0.5,0.5],[0.4,0.5],[0.5,0.4],
                      [-0.5,-0.5],[-0.4,-0.5],[-0.5,-0.4],
                      [-0.5,0.5],[-0.4,0.5],[-0.5,0.4]])
    distance = pf.distance
    n_clusters = 3
    centroids_index, centroids_value = _init_random(X, n_clusters, distance)
    plt.scatter(X[:,0].numpy(), X[:,1].numpy(), label="point")
    plt.scatter(centroids_value[:,0].numpy(), centroids_value[:,1].numpy(), label="initial centroids")
    plt.title("Test random initialisation")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    # execute all the test
    _test_init_kmeans_pp()
    _test_init_random()


