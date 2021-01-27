'''
Linear euclidean classifiers definition
'''
import torch
from torch import nn
from torch import optim
from sklearn import svm
import tqdm


'''
A linear classifier optimizing a cross entropy criterion
'''
class EuclideanClassifierBinaryCrossEntropy(object):
    def __init__(self, n_classes):
        self._n_c = n_classes

    def fit(self, X, Y=None, iteration=1000):
        self.model = nn.Linear(X.size(-1), self._n_c)
        self.model = self.model.to(X.device)
        optimizer  = optim.Adam(self.model.parameters(), lr=1e-2)
        criterion = torch.nn.BCEWithLogitsLoss()
        pb = tqdm.trange(iteration)
        for i in pb:
            optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, Y.float())
            loss.backward()
            optimizer.step()
            pb.set_postfix({"loss": loss.mean().item()})

    def probs(self, z):
        with torch.no_grad():
            return self.model(z).sigmoid()


'''
A linear SVM using sklearn
'''
class SKLearnSVM(object):
    def __init__(self, n_classes):
        self._n_c = n_classes
    def fit(self, X, Y=None, iteration=5000):
        self.model = svm.LinearSVC()
        self.model.fit(X.numpy(), Y.max(-1)[-1].numpy())

    def probs(self, z):
        predicted = self.model.predict(z.numpy())
        res = torch.zeros(len(z), self._n_c)
        for i, l in enumerate(predicted):
            res[i][l] = 1
        return res

