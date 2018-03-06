from asyncio import Lock
from utils import random_key


class UpdateException(Exception):
    pass


class UpdateInProgress(UpdateException):
    pass


class UpdateNotInProgress(UpdateException):
    pass


class UpdateManager(object):
    def __init__(self, name=None):
        self.lock = Lock()
        self.name = name or random_key(6)
        self.loss_history = []
        self.n_updates = 0
        self._reset_state()

    def _reset_state(self):
        self.update_name = "update_{}_{:05d}".format(self.name, self.n_updates)
        self.clients = set()
        self.client_responses = dict()
        self.update_meta = None

    @property
    def in_progress(self):
        return self.lock.locked()

    @property
    def clients_left(self):
        return len(self.clients) - len(self.client_responses)

    def __len__(self):
        return self.in_progress * len(self.clients)

    async def start_update(self, **update_meta):
        print("starting update")
        if self.in_progress:
            raise UpdateInProgress
        self._reset_state()
        await self.lock.acquire()
        self.update_meta = update_meta

    def end_update(self):
        self.lock.release()
        self.n_updates += 1
        return self.client_responses

    def client_start(self, client_id):
        if not self.in_progress:
            raise UpdateNotInProgress
        self.clients.add(client_id)

    def client_end(self, client_id, response):
        if not self.in_progress:
            raise UpdateNotInProgress
        self.client_responses[client_id] = response
        print("Update finished: {} [{}/{}]".format(
            client_id,
            len(self.client_responses),
            len(self.clients))
        )
