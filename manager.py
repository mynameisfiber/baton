import pickle
from aiohttp import web

from client_manager import ClientManager

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
        self.client_manager = ClientManager(name, app, client_ttl)
        self.register_handlers()

        self._n_updates = 0
        self._update_state = {}
        self._update_loss_history = []

    def register_handlers(self):
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

    async def start_round(self, n_epoch):
        if self._update_state.get('in_progress', False):
            raise UpdateInProgress
        update_name = "update_{}_{:05d}".format(self.name, self._n_updates)
        print("Starting update:", update_name)
        if not len(self.client_manager):
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
        data = {
            'state_dict': self.model.state_dict(),
            'update_name': update_name,
            'n_epoch': n_epoch,
        }
        data_pkl = pickle.dumps(data)
        result = await self.client_manager.notify_clients(
            'round_start',
            http_method='POST',
            data=data_pkl
        )
        for client_id, response in result:
            if response:
                self._update_state['clients'].add(client_id)
        if not self._update_state['clients']:
            print("No clients working on round... ending")
            self.end_round()
        return dict(result)

    async def update(self, request):
        client_id = self.client_manager.verify_request(request)
        body = await request.read()
        data = pickle.loads(body)

        update_name = data['update_name']

        if (not self._update_state.get('in_progress', False) or
                update_name != self._update_state['name']):
            return web.json_response({'error': "Wrong Update"}, status=410)

        self._update_state['clients_done'].add(client_id)
        self._update_state['client_responses'][client_id] = data
        self.client_manager[client_id]['last_update'] = update_name
        self.client_manager[client_id]['num_updates'] += 1
        print("Update finished: {} [{}/{}]".format(
            client_id,
            self.client_manager[client_id]['remote'],
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
