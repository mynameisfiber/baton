import torch
from torch import nn
from torch import optim
from torch import autograd
from aiohttp import web

from utils import EpochProgress
from manager import Manager
from worker import ExperimentWorker

import random


class Model(nn.Module):
    name = "lineartest"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 1)

    def forward(self, X):
        X = self.fc1(X)
        return X

    def __hash__(self):
        return hash(tuple((k, *v.shape) for k, v in self.state_dict().items()))

    def train(self, X, y, n_epoch=32, lr=0.001, batch_size=32, verbose=True):
        X = autograd.Variable(X)
        y = autograd.Variable(y)
        loss = nn.MSELoss()
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


class LinearTestWorker(ExperimentWorker):
    def get_data(self):
        n = random.randint(5, 20)
        p = torch.Tensor((11, 5, 3, 2, 5, 6, 2, 7, 8, 1))
        X = torch.randn(32*n, 10)
        y = (p * X).sum(1)
        print(y.shape)
        return (X, y), n*32


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, help="process's role, must be 'worker' or 'manager'")
    parser.add_argument("--manager-host", type=str, help="hostname of the experiment manager")
    parser.add_argument("--manager-port", type=int, default=8080,
                        help="port the manager is listening on")
    parser.add_argument("--port", type=int, default=8081,
                        help="communication port for this process")
    args = parser.parse_args()

    app = web.Application()

    if args.role == 'manager':
        app = web.Application()
        manager = Manager(app)
        model = Model()
        manager.register_experiment(model)
    elif args.role == 'worker':
        model = Model()
        manager_host_name = args.manager_host + ":" + str(args.manager_port)
        worker = LinearTestWorker(app, model, manager_host_name, port=args.port)
    web.run_app(app, port=args.port)
