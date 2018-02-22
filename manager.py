import asyncio
import aiohttp
from aiohttp import web
from urllib.parse import urljoin
import pickle

from experiments import Model


class Manager(object):
    def __init__(self, app):
        self.app = app
        self.experiments = []

    def register_experiment(self, model, name=None):
        name = name or getattr(model, 'name', hash(model))
        experiment = Experiment(name, app, model)
        self.experiments.append(experiment)


class Experiment(object):
    def __init__(self, name, app, model):
        self.name = name
        self.model = model
        self.app = app
        self.clients = {}
        self._session = aiohttp.ClientSession()
        self.register_handlers()

        self._n_updates = 0
        self._update_state = {}
        self._update_loss_history = []

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
            '/{}/loss_history'.format(self.name),
            self.get_loss_history,
        )

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
        status = await self.start_round(n_epoch)
        return web.json_response(status)

    async def start_round(self, n_epoch):
        if self._update_state.get('in_progress', False):
            raise Exception("Update already in progress")
        update_name = "update_{}_{:05d}".format(self.name, self._n_updates)
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
        client_id = "client_{}_{:08d}".format(self.name, len(self.clients))
        state = {
            'client_id': client_id,
        }
        url = "http://{}:{}/{}/".format(request.remote, data['port'],
                                        self.name)
        self.clients[client_id] = {
            "remote": request.remote,
            "port": data['port'],
            "url": url,
            "last_update": None,
            "num_updates": 0,
        }
        print("Registered client:", client_id)
        return web.json_response(state)

    async def update(self, request, force_end=False):
        body = await request.read()
        data = pickle.loads(body)

        client_id = data['client_id']
        update_name = data['update_name']

        if (not self._update_state.get('in_progress', False) or
                update_name != self._update_state['name']):
            return web.json_response({'error': "Wrong Update"}, status=410)

        self._update_state['clients_done'].add(client_id)
        self._update_state['client_responses'][client_id] = data
        self.clients[client_id]['last_update'] = update_name
        self.clients[client_id]['num_updates'] += 1
        print("Update finished: {} [{}/{}]".format(
            client_id,
            len(self._update_state['clients_done']),
            len(self._update_state['clients']))
        )

        if (self._update_state['clients'] == self._update_state['clients_done']
                or force_end):
            self._update_state['in_progress'] = False
            self._n_updates += 1
            datas = self._update_state['client_responses'].values()
            N = sum(d['n_samples'] for d in datas)
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
        return web.json_response("OK")


if __name__ == "__main__":
    app = web.Application()
    manager = Manager(app)
    model = Model()
    manager.register_experiment(model)
    web.run_app(app, port=8080)
