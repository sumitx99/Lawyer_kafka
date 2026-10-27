"""S3-compatible object access, targeting SeaweedFS without AWS services."""

import asyncio
import boto3
from botocore.config import Config

from .secrets_control import Settings, get_settings


class Storage:
    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        self.bucket = settings.storage_bucket
        self.client = boto3.client(
            's3', endpoint_url=settings.storage_endpoint_url,
            aws_access_key_id=settings.storage_access_key.get_secret_value(),
            aws_secret_access_key=settings.storage_secret_key.get_secret_value(),
            region_name=settings.storage_region,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                          connect_timeout=5, read_timeout=30,
                          retries={'max_attempts': 3, 'mode': 'standard'}),
        )

    @staticmethod
    def _key(key):
        if not key or key.startswith('/') or '..' in key.split('/'):
            raise ValueError('An object key must be a nonempty relative path without parent traversal.')
        return key

    async def put_bytes(self, key: str, data: bytes, content_type='application/octet-stream'):
        await asyncio.to_thread(self.client.put_object, Bucket=self.bucket, Key=self._key(key),
                                Body=data, ContentType=content_type)

    async def get_bytes(self, key: str) -> bytes:
        key = self._key(key)

        def read():
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            with response['Body'] as body:
                return body.read()

        return await asyncio.to_thread(read)

    async def delete(self, key: str):
        await asyncio.to_thread(self.client.delete_object, Bucket=self.bucket, Key=self._key(key))

    async def ping(self):
        await asyncio.to_thread(self.client.head_bucket, Bucket=self.bucket)
        return True

    async def close(self):
        await asyncio.to_thread(self.client.close)
