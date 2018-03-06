import pickle
from aiohttp import web

from client_manager import ClientManager
from update_manager import UpdateManager, UpdateException

from utils import json_clean


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
        self.update_manager = UpdateManager(name)
        self.register_handlers()

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
        except UpdateException:
            return web.json_response({'err': "Update already in progress"},
                                     status=423)
        return web.json_response(status)

    async def trigger_end_round(self, request):
        self.end_round()
        return web.json_response(json_clean(self._update_state))

    async def start_round(self, n_epoch):
        await self.update_manager.start_update(n_epoch=n_epoch)
        update_name = self.update_manager.update_name
        print("Starting update:", update_name)
        if not len(self.client_manager):
            print("No clients. Aborting update.")
            return []
        data = {
            'state_dict': self.model.state_dict(),
            'update_name': update_name,
            'n_epoch': n_epoch,
        }
        result = await self.client_manager.notify_clients(
            'round_start',
            http_method='POST',
            data=pickle.dumps(data)
        )
        for client_id, response in result:
            if response:
                self.update_manager.client_start(client_id)
        if not self.update_manager:
            print("No clients working on round... ending")
            self.end_round()
        return dict(result)

    async def update(self, request):
        client_id = self.client_manager.verify_request(request)
        body = await request.read()
        data = pickle.loads(body)
        update_name = data['update_name']

        if (not self.update_manager.in_progress or
                update_name != self.update_manager.update_name):
            return web.json_response({'error': "Wrong Update"}, status=410)

        self.update_manager.client_end(client_id, data)
        self.client_manager[client_id]['last_update'] = update_name
        self.client_manager[client_id]['num_updates'] += 1

        if not self.update_manager.clients_left:
            self.end_round()
        return web.json_response("OK")

    def end_round(self):
        if not self.update_manager.in_progress:
            return
        update_name = self.update_manager.update_name
        print("Finishing update:", update_name)
        datas = self.update_manager.end_update()
        N = sum(d['n_samples'] for d in datas.values())
        if not N:
            print("No responses for update:", update_name)
            return
        for key, value in self.model.state_dict().items():
            weight_sum = (d['state_dict'][key] * d['n_samples']
                          for d in datas.values())
            value[:] = sum(weight_sum) / N
        for epoch in range(self.update_manager.update_meta['n_epoch']):
            epoch_loss = sum(d['loss_history'][epoch]*d['n_samples']
                             for d in datas.values())
            self.update_manager.loss_history.append(epoch_loss / N)
        print("Finished update:", update_name)
        print("Final Loss:", self.update_manager.loss_history[-1])
