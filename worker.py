from urllib.parse import urljoin
import torch
import random
import pickle

import asyncio
import aiohttp
from aiohttp import web

from experiments import Model


class ExperimentWorker(object):
    def __init__(self, name, app, model, manager, port=8080):
        self.name = name
        self.model = model
        self.app = app
        self.port = port
        self.manager = manager
        self.manager_url = "http://{}/{}/".format(manager, name)
        self._session = aiohttp.ClientSession()
        self.register_handlers()
        self.n_updates = 0
        self.last_update = None
        asyncio.ensure_future(self.register_with_manager())

    async def register_with_manager(self):
        url = urljoin(self.manager_url, 'register')
        data = {'port': self.port}
        print("registering with:",url)
        async with self._session.get(url, json=data) as resp:
            response = await resp.json()
            self.client_id = response['client_id']
            print("I am now:", self.client_id)

    def register_handlers(self):
        self.app.router.add_post(
            '/{}/round_start'.format(self.name),
            self.round_start
        )

    async def round_start(self, request):
        body = await request.read()
        data = pickle.loads(body)
        if data['client_id'] != self.client_id:
            return web.json_response({"err": "Wrong Client"}, status=404)
        self.last_update = update_name = data['update_name']
        self.model.load_state_dict(data['state_dict'])
        asyncio.ensure_future(self._run_round(update_name))
        return web.json_response("OK")

    async def _run_round(self, update_name):
        data, n_samples = self.get_data()
        self.model.train(*data)
        await self.report_update(update_name, n_samples)

    async def report_update(self, update_name, n_samples):
        url = urljoin(self.manager_url, 'update')
        state = {}
        state['state_dict'] = self.model.state_dict()
        state['client_id'] = self.client_id
        state['n_samples'] = n_samples
        state['update_name'] = update_name
        data = pickle.dumps(state)
        async with self._session.post(url, data=data) as resp:
            if resp.status == 200:
                self.n_updates += 1

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
    worker = ExperimentWorker("exp1", app, model, 'localhost:8080', port)
    web.run_app(app, port=port)
