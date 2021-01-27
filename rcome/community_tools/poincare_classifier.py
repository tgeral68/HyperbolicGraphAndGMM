'''
A logistic regression based classifier in hyperbolic space inspired from
Hyperbolic Neural Network (Ganea et al 2018)
'''
import torch
import tqdm

from torch import optim
from torch.utils.data import DataLoader

from rcome.optim_tools import optimizer as ph
from rcome.function_tools import poincare_module as pm
from rcome.data_tools import corpora_tools

def hinge_loss(x,y):
    v = x*y
    return v[v>0].sum()

class PoincareClassifier(object):
    def __init__(self, n_classes, criterion="BCE"):
        self._n_c = n_classes
        self._criterion = criterion
    def fit(self, X, Y=None, iteration=50):
        Y = Y.double()
        self.model = pm.PoincareMLR(X.size(-1), Y.size(-1))
        if(X.is_cuda):
            self.model.cuda()
        optimizer_euclidean = optim.Adam(self.model.euclidean_parameters(), lr=8e-3)
        optimizer_hyperbolic = ph.PoincareRAdam(self.model.poincare_parameters(), lr=5e-3)

        if(self._criterion == "BCE"):
            criterion = torch.nn.BCEWithLogitsLoss()

        if(self._criterion == "HINGE"):
            criterion = hinge_loss
            Y = Y.clone()
            Y[Y==0] = -1


        zip_corpora = corpora_tools.zip_datasets(corpora_tools.from_indexable(X),corpora_tools.from_indexable(Y))
        dataloader = DataLoader(zip_corpora, 
                                batch_size=500, 
                                shuffle=True
                            )
        progress_bar = tqdm.trange(iteration)
        for i in progress_bar:
            tloss = 0
            for x, y in dataloader:
                optimizer_euclidean.zero_grad()
                optimizer_hyperbolic.zero_grad()

                pred = self.model(x)

                loss = criterion(pred,y)
                tloss += loss.item()

                loss.backward()

                optimizer_euclidean.step()
                optimizer_hyperbolic.step()
            progress_bar.set_postfix({"loss":tloss, "max_pred":pred.max().item()})
    def probs(self, z):
        with torch.no_grad():
            return self.model(z).sigmoid()
    def predict(self, z):
        with torch.no_grad():
            return self.model(z).max(-1)[1]

