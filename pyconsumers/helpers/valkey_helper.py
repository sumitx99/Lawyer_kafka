"""Namespaced caches and token-owned locks; durable job state remains in PostgreSQL."""

import json
import secrets

from valkey.asyncio import Valkey

from .secrets_control import Settings, get_settings


class Cache:
    RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"

    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        self.client = Valkey(host=settings.valkey_host, port=settings.valkey_port,
                             password=settings.valkey_password.get_secret_value(),
                             decode_responses=True, socket_connect_timeout=5, socket_timeout=5)

    @staticmethod
    def key(project_id, kind, identity):
        return f'rag:v1:{project_id}:{kind}:{identity}'

    async def set_json(self, key, value, ttl=300):
        if ttl <= 0:
            raise ValueError('Cache TTL must be positive.')
        await self.client.set(key, json.dumps(value), ex=ttl)

    async def get_json(self, key):
        value = await self.client.get(key)
        return json.loads(value) if value is not None else None

    async def acquire_lock(self, key, ttl=60):
        if ttl <= 0:
            raise ValueError('Lock TTL must be positive.')
        token = secrets.token_hex(24)
        return token if await self.client.set(key, token, nx=True, ex=ttl) else None

    async def release_lock(self, key, token):
        return bool(await self.client.eval(self.RELEASE, 1, key, token))

    async def close(self):
        await self.client.aclose()
