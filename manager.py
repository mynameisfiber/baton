import asyncio
import aiohttp
import pickle
from aiohttp import web
from urllib.parse import urljoin
from datetime import datetime, timedelta

from utils import random_key
from utils import PeriodicTask
from utils import json_clean


class UpdateInProgress(Exception):
    pass


class Manager(object):
    def __init__(self, app):
        self.app = app
        self.experiments = []

    def register_experiment(self, model, name=None):
        name = name or getattr(model, 'name', hash(model))
        experiment = Experiment(name, self.app, model)
        self.experiments.append(experiment)


class Experiment(object):
    def __init__(self, name, app, model, client_ttl=300):
        self.name = name
        self.model = model
        self.app = app
        self.clients = {}
        self._session = aiohttp.ClientSession()
        self.register_handlers()

        self.client_ttl = timedelta(seconds=client_ttl)
        self._n_updates = 0
        self._update_state = {}
        self._update_loss_history = []
        self._stale_manager = PeriodicTask(self.cull_clients,
                                           client_ttl//2).start()

    def register_handlers(self):
        self.app.router.add_get(
            '/{}/register'.format(self.name),
            self.register
        )
        self.app.router.add_post(
            '/{}/update'.format(self.name),
            self.update,
        )
        self.app.router.add_get(
            '/{}/start_round'.format(self.name),
            self.trigger_start_round,
        )
        self.app.router.add_get(
            '/{}/end_round'.format(self.name),
            self.trigger_end_round,
        )
        self.app.router.add_get(
            '/{}/loss_history'.format(self.name),
            self.get_loss_history,
        )
        self.app.router.add_get(
            '/{}/clients'.format(self.name),
            self.get_clients,
        )
        self.app.router.add_get(
            '/{}/heartbeat'.format(self.name),
            self.heartbeat,
        )

    async def heartbeat(self, request):
        # TODO: special client handler object to do all this and culling?
        data = await request.json()
        client_id = data['client_id']
        key = data['key']
        if client_id not in self.clients:
            print("Invalid heartbeat: Unknown Client:", client_id)
            return web.json_response({'err': "Invalid Client"}, status=401)
        elif self.clients[client_id]['key'] != key:
            print(key)
            print(self.clients[client_id])
            print("Invalid heartbeat: Invalid Key:", client_id)
            return web.json_response({'err': "Invalid Key"}, status=401)
        self.clients[client_id]['last_heartbeat'] = datetime.now()
        return web.json_response("OK")

    async def get_clients(self, request):
        data = [json_clean(client)
                for client in self.clients.values()]
        return web.json_response(data)

    async def get_loss_history(self, request):
        return web.json_response(self._update_loss_history)

    async def trigger_start_round(self, request):
        try:
            n_epoch = int(request.query['n_epoch'])
        except KeyError:
            n_epoch = 32
        except ValueError:
            return web.json_response({"err": "Invalid Epoch Value"},
                                     status=400)
        try:
            status = await self.start_round(n_epoch)
        except UpdateInProgress:
            return web.json_response({'err': "Update already in progress"},
                                     status=423)
        return web.json_response(status)

    async def trigger_end_round(self, request):
        self.end_round()
        return web.json_response(json_clean(self._update_state))

    async def cull_clients(self):
        now = datetime.now()
        stale_clients = set()
        for client_id, client in self.clients.items():
            if (now - client['last_heartbeat']) > self.client_ttl:
                stale_clients.add(client_id)
        for stale_client in stale_clients:
            print("Removing stale client:", stale_client)
            self.clients.pop(stale_client)

    async def start_round(self, n_epoch):
        if self._update_state.get('in_progress', False):
            raise UpdateInProgress
        update_name = "update_{}_{:05d}".format(self.name, self._n_updates)
        print("Starting update:", update_name)
        await self.cull_clients()
        if not len(self.clients):
            print("No clients. Aborting update.")
            return []
        self._update_state = {
            'name': update_name,
            'clients': set(),
            'clients_done': set(),
            'client_responses': dict(),
            'n_epoch': n_epoch,
            'in_progress': True,
        }
        return await asyncio.gather(
            *[self._notify_round_start(update_name, c, n_epoch)
              for c in self.clients]
        )

    async def _notify_round_start(self, update_name, client_id, n_epoch):
        url = urljoin(self.clients[client_id]['url'], 'round_start')
        data = {
            'state_dict':  self.model.state_dict(),
            'update_name':  update_name,
            'client_id':  client_id,
            'n_epoch':  n_epoch,
        }
        data_pkl = pickle.dumps(data)
        try:
            async with self._session.post(url, data=data_pkl) as resp:
                if resp.status == 200:
                    self._update_state['clients'].add(client_id)
                    return True
                elif resp.status == 404:
                    self.clients.pop(client_id)
        except aiohttp.client_exceptions.ClientConnectorError:
            self.clients.pop(client_id)
        return False

    async def register(self, request):
        data = await request.json()
        remote = request.remote
        client_id = "client_{}_{}".format(self.name, random_key(6))
        key = random_key()
        state = {
            'client_id': client_id,
            'key': key,
        }
        url = "http://{}:{}/{}/".format(request.remote, data['port'],
                                        self.name)
        self.clients[client_id] = client = {
            "key": key,
            "client_id": client_id,
            "remote": remote,
            "port": data['port'],
            "last_heartbeat": datetime.now(),
            "url": url,
            "last_update": None,
            "num_updates": 0,
        }
        print("Registered client:", client_id, client['remote'], client['port'])
        return web.json_response(state)

    async def update(self, request):
        body = await request.read()
        data = pickle.loads(body)

        client_id = data['client_id']
        client_key = data['key']
        update_name = data['update_name']

        if client_key != self.clients[client_id]['key']:
            return web.json_response({'error': "Invalid Client Key"},
                                     status=401)
        if (not self._update_state.get('in_progress', False) or
                update_name != self._update_state['name']):
            return web.json_response({'error': "Wrong Update"}, status=410)

        self._update_state['clients_done'].add(client_id)
        self._update_state['client_responses'][client_id] = data
        self.clients[client_id]['last_update'] = update_name
        self.clients[client_id]['num_updates'] += 1
        print("Update finished: {} [{}/{}]".format(
            client_id,
            self.clients[client_id]['remote'],
            len(self._update_state['clients_done']),
            len(self._update_state['clients']))
        )

        if self._update_state['clients'] == self._update_state['clients_done']:
            self.end_round()
        return web.json_response("OK")

    def end_round(self):
        if not self._update_state.get('in_progress', False):
            return
        print("Finishing update:", self._update_state['name'])
        self._update_state['in_progress'] = False
        self._n_updates += 1
        datas = self._update_state['client_responses'].values()
        N = sum(d['n_samples'] for d in datas)
        if not N:
            print("No responses for update:", self._update_state['name'])
            return
        for key, value in self.model.state_dict().items():
            weight_sum = (d['state_dict'][key] * d['n_samples']
                          for d in datas)
            value[:] = sum(weight_sum) / N
        for epoch in range(self._update_state['n_epoch']):
            epoch_loss = sum(d['loss_history'][epoch]*d['n_samples']
                             for d in datas)
            self._update_loss_history.append(epoch_loss / N)
        print("Finished update:", self._update_state['name'])
        print("Final Loss:", self._update_loss_history[-1])
