"""Async SQLAlchemy/asyncpg sessions with explicit lifetime and transaction boundaries."""

from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .secrets_control import Settings, get_settings


class Database:
    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        self.engine = create_async_engine(settings.database_url, pool_pre_ping=True,
                                          pool_size=settings.db_pool_size,
                                          max_overflow=settings.db_max_overflow,
                                          hide_parameters=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def transaction(self):
        async with self.sessions() as session:
            async with session.begin():
                yield session

    async def ping(self):
        async with self.engine.connect() as connection:
            return await connection.scalar(text('SELECT 1')) == 1

    async def close(self):
        await self.engine.dispose()
