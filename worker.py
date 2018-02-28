from urllib.parse import urljoin
import pickle

import asyncio
import aiohttp
from aiohttp import web

from utils import PeriodicTask
from utils import ensure_no_collision


class ExperimentWorker(object):
    def __init__(self, app, model, manager, name=None, port=8080,
                 heartbeat_time=60, worker_host=None):
        self.name = name or getattr(model, 'name', hash(model))
        self.model = model
        self.app = app
        self.port = port
        self.worker_host = worker_host
        self.manager = manager
        self.manager_url = "http://{}/{}/".format(manager, self.name)
        self.__session = None
        self.register_handlers()
        self.n_updates = 0
        self.update_in_progress = False
        self.last_update = None
        self.client_id = None
        self.key = None
        self.heartbeat_time = heartbeat_time
        self._heartbeat_manager = None
        self._heartbeat_lock = asyncio.Lock()
        asyncio.ensure_future(self.register_with_manager())

    @property
    def _session(self):
        if not self.__session:
            self.__session = aiohttp.ClientSession()
        return self.__session

    @ensure_no_collision
    async def register_with_manager(self):
        url = urljoin(self.manager_url, 'register')
        data = {'url': self.worker_host, 'port': self.port}
        print("registering with:", url)
        async with self._session.get(url, json=data) as resp:
            response = await resp.json()
            self.client_id = response['client_id']
            self.key = response['key']
            print("I am now:", self.client_id)
            if self._heartbeat_manager is not None:
                await self._heartbeat_manager.stop()
            self._heartbeat_manager = PeriodicTask(
                self.heartbeat,
                self.heartbeat_time,
            ).start()

    @ensure_no_collision
    async def heartbeat(self):
        timeout = 1
        while True:
            url = urljoin(self.manager_url, 'heartbeat')
            data = {
                'client_id': self.client_id,
                'key': self.key,
            }
            try:
                async with self._session.get(url, json=data) as resp:
                    if resp.status == 200:
                        print(".", end='', flush=True)
                        return
                    elif resp.status == 401:
                        print("Reregistering with manager")
                        return await self.register_with_manager()
            except aiohttp.client_exceptions.ClientConnectorError:
                pass
            print("Could not connect to master, waiting:", timeout)
            await asyncio.sleep(timeout)
            # TODO: better backoff
            timeout *= 2

    def register_handlers(self):
        self.app.router.add_post(
            '/{}/round_start'.format(self.name),
            self.round_start
        )

    async def round_start(self, request):
        if self.update_in_progress:
            return web.json_response({"err": "Update in Progress"},
                                     status=409)
        body = await request.read()
        data = pickle.loads(body)
        if (request.query['client_id'] != self.client_id or
                request.query['key'] != self.key):
            asyncio.ensure_future(self.register_with_manager())
            return web.json_response({"err": "Wrong Client"}, status=404)
        self.last_update = update_name = data['update_name']
        self.model.load_state_dict(data['state_dict'])
        n_epoch = data['n_epoch']
        asyncio.ensure_future(self._run_round(update_name, n_epoch))
        return web.json_response("OK")

    async def _run_round(self, update_name, n_epoch):
        data, n_samples = self.get_data()
        loss_history = self.model.train(*data, n_epoch=n_epoch)
        await self.report_update(update_name, n_samples, loss_history)

    async def report_update(self, update_name, n_samples, loss_history):
        url = urljoin(self.manager_url, 'update')
        url += "?client_id={}&key={}".format(self.client_id, self.key)  # TODO: fix this
        state = {
            'state_dict':  self.model.state_dict(),
            'n_samples':  n_samples,
            'update_name':  update_name,
            'loss_history':  loss_history,
        }
        data = pickle.dumps(state)
        async with self._session.post(url, data=data) as resp:
            if resp.status == 200:
                self.n_updates += 1
            elif resp.status == 401:
                await self.register_with_manager()
            elif resp.status == 410:
                print("Sent wrong update")

    def get_data(self):
        raise NotImplementedError
