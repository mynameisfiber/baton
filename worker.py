from urllib.parse import urljoin
import torch
import random
import pickle

import asyncio
import aiohttp
from aiohttp import web

from experiments import Model
from utils import PeriodicTask


class ExperimentWorker(object):
    def __init__(self, app, model, manager, name=None, port=8080, heartbeat_time=5):
        self.name = name or getattr(model, 'name', hash(model))
        self.model = model
        self.app = app
        self.port = port
        self.manager = manager
        self.manager_url = "http://{}/{}/".format(manager, self.name)
        self._session = aiohttp.ClientSession()
        self.register_handlers()
        self.n_updates = 0
        self.update_in_progress = False
        self.last_update = None
        self.client_id = None
        self.key = None
        self.heartbeat_time = heartbeat_time
        self._heartbeat_manager = None
        asyncio.ensure_future(self.register_with_manager())

    async def register_with_manager(self):
        url = urljoin(self.manager_url, 'register')
        data = {'port': self.port}
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

    async def heartbeat(self, timeout=1):
        url = urljoin(self.manager_url, 'heartbeat')
        data = {
            'client_id': self.client_id,
            'key': self.key,
        }
        try:
            async with self._session.get(url, json=data) as resp:
                if resp.status == 200:
                    print(".", end='')
                    return
                elif resp.status == 401:
                    print("Reregistering with manager")
                    await self.register_with_manager()
                    return
        except aiohttp.client_exceptions.ClientConnectorError:
            pass
        print("Could not connect to master, waiting:", timeout)
        await asyncio.sleep(timeout)
        await self.heartbeat(timeout*2)
        return

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
        if data['client_id'] != self.client_id:
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
        state = {
            'state_dict':  self.model.state_dict(),
            'client_id':  self.client_id,
            'n_samples':  n_samples,
            'update_name':  update_name,
            'loss_history':  loss_history,
            'key': self.key,
        }
        data = pickle.dumps(state)
        async with self._session.post(url, data=data) as resp:
            if resp.status == 200:
                self.n_updates += 1
            elif resp.status == 401:
                await self.register_with_manager()

    def get_data(self):
        n = random.randint(5, 20)
        X = torch.randn(32*n, 28*28)
        _, y = torch.randn(32*n, 10).max(1)
        return (X, y), n*32


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1])
    app = web.Application()
    model = Model()
    worker = ExperimentWorker(app, model, 'localhost:8080', port=port)
    web.run_app(app, port=port)
