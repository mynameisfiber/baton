from tqdm import tqdm
from collections import Iterator


class EpochProgress(Iterator):
    def __init__(self, epoch, _iter, verbose=True):
        self.verbose = verbose
        if verbose:
            self.pbar = tqdm(_iter, desc="Epoch {}".format(epoch))
        else:
            self.pbar = _iter
        self.pbar_iter = iter(self.pbar)
        self.loss = 0
        self.N = 0

    def __next__(self):
        self.N += 1
        return next(self.pbar_iter)

    def update_loss(self, loss):
        i = self.N
        self.loss *= i / (i + 1)
        self.loss += float(loss) / i
        if self.verbose is True:
            self.pbar.set_postfix({"loss": "{:0.4f}".format(self.loss)})
