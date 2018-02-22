import asyncio
from contextlib import suppress

from tqdm import tqdm
from collections import Iterator
import random
import string


def random_key(length=32):
    return "".join(random.sample(string.ascii_letters, length))


class PeriodicTask(object):
    def __init__(self, func, time):
        self.func = func
        self.time = time
        self.is_started = False
        self._task = None

    def start(self):
        if not self.is_started:
            self.is_started = True
            # Start task to call func periodically:
            self._task = asyncio.ensure_future(self._run())
        return self

    async def stop(self):
        if self.is_started:
            self.is_started = False
            # Stop task and await it stopped:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self):
        while True:
            await asyncio.sleep(self.time)
            await self.func()


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
