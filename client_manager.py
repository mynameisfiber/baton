from aiohttp import web
import aiohttp
import asyncio

from datetime import datetime
from datetime import timedelta
from urllib.parse import urljoin

from utils import PeriodicTask
from utils import random_key
from utils import json_clean


class ClientManager(object):
    def __init__(self, name, app, client_ttl=300):
        self.name = name
        self.app = app
        self.__session = None

        self.client_ttl = timedelta(seconds=client_ttl)
        self.clients = {}
        self.register_handlers()
        self._stale_manager = PeriodicTask(self.cull_clients,
                                           client_ttl//2).start()

    def __len__(self):
        return len(self.clients)

    @property
    def _session(self):
        if not self.__session:
            self.__session = aiohttp.ClientSession()
        return self.__session

    async def notify_clients(self, client_method, http_method='GET',
                             client_callback=None, notify_callback=None,
                             **kwargs):
        await self.cull_clients()
        result = await asyncio.gather(
            *[self.notify_client(c, client_method, http_method=http_method,
                                 callback=client_callback,
                                 **kwargs)
              for c in self.clients]
        )
        if notify_callback is not None:
            return await notify_callback(result)
        return result

    async def notify_client(self, client_id, client_method, http_method='GET',
                            callback=None, **kwargs):
        url = urljoin(self.clients[client_id]['url'], client_method)
        url += "?client_id={}&key={}".format(client_id, self.clients[client_id]['key'])  # TODO: fix this
        result = False
        try:
            async with self._session.request(http_method, url, **kwargs) as resp:
                if resp.status == 200:
                    result = True
                elif resp.status == 404:
                    self.clients.pop(client_id)
        except aiohttp.client_exceptions.ClientConnectorError:
            self.clients.pop(client_id)
        if callback is not None:
            await callback(client_id, result)
        return client_id, result

    def register_handlers(self):
        self.app.router.add_get(
            '/{}/register'.format(self.name),
            self.register
        )
        self.app.router.add_get(
            '/{}/clients'.format(self.name),
            self.get_clients,
        )
        self.app.router.add_get(
            '/{}/heartbeat'.format(self.name),
            self.heartbeat,
        )

    def __getitem__(self, key):
        return self.clients[key]

    def __setitem__(self, key, value):
        return self.clients.__setitem__(key, value)

    async def register(self, request):
        data = await request.json()
        remote = request.remote
        client_id = "client_{}_{}".format(self.name, random_key(6))
        key = random_key()
        state = {
            'client_id': client_id,
            'key': key,
        }
        if data.get('url'):
            url = data['url']
        else:
            url = "http://{}:{}/{}/".format(remote, data['port'],
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
        print("Registered client:", client_id, client['url'], client['port'])
        return web.json_response(state)

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

    async def cull_clients(self):
        now = datetime.now()
        stale_clients = set()
        for client_id, client in self.clients.items():
            if (now - client['last_heartbeat']) > self.client_ttl:
                stale_clients.add(client_id)
        for stale_client in stale_clients:
            print("Removing stale client:", stale_client)
            self.clients.pop(stale_client)

    async def get_clients(self, request):
        data = [json_clean(client)
                for client in self.clients.values()]
        return web.json_response(data)

    def verify_request(self, request):
        client_id = request.query['client_id']
        client_key = request.query['key']
        if (client_id not in self.clients or
                client_key != self.clients[client_id]['key']):
            raise web.HTTPUnauthorized()
        return client_id
