import torch
from torch import nn
from torch import optim
from torch.nn import functional
from torch import autograd

import random
from copy import deepcopy
from multiprocessing import Pool

from utils import EpochProgress


class Model(nn.Module):
    name = "lineartest"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = functional.tanh(self.fc1(X))
        X = functional.tanh(self.fc2(X))
        X = self.softmax(self.fc3(X))
        return X

    def __hash__(self):
        return hash(tuple((k, *v.shape) for k, v in self.state_dict().items()))

    def train(self, X, y, n_epoch=32, lr=0.001, batch_size=32, verbose=True):
        X = autograd.Variable(X)
        y = autograd.Variable(y)
        loss = nn.CrossEntropyLoss()
        idxs = torch.randperm(X.shape[0])
        optimizer = optim.SGD(self.parameters(), lr=lr)
        loss_history = []
        for epoch in range(n_epoch):
            batch_iter = EpochProgress(epoch, torch.split(idxs, batch_size),
                                       verbose=verbose)
            for batch_idxs in batch_iter:
                optimizer.zero_grad()
                X_batch = X[batch_idxs]
                y_batch = y[batch_idxs]
                output = self(X_batch)
                loss_batch = loss(output, y_batch)
                batch_iter.update_loss(loss_batch)
                loss_batch.backward()
                optimizer.step()
            loss_history.append(batch_iter.loss)
        return loss_history


def client_update(state_dict):
    n = random.randint(5, 20)
    print(n)
    X = torch.randn(32*n, 28*28)
    _, y = torch.randn(32*n, 10).max(1)
    model = Model()
    model.load_state_dict(state_dict)
    model.train(X, y, verbose=False)
    return model.state_dict(), n*32


def fedavg(num_clients):
    model = Model()
    with Pool(num_clients) as client_pool:
        for i in range(10):
            print("Epoch:", i)
            params = [deepcopy(model.state_dict())
                      for _ in range(num_clients)]
            results = client_pool.map(client_update, params)
            N = sum(r[1] for r in results)
            for key, value in model.state_dict().items():
                value[:] = sum(r[0][key] * r[1] for r in results) / N


if __name__ == "__main__":
    fedavg(4)
