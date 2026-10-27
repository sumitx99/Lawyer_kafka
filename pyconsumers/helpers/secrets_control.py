"""Validated runtime settings. No clients or network connections are created here."""

from functools import lru_cache
import os
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore', case_sensitive=False, hide_input_in_errors=True)

    postgres_host: str = '127.0.0.1'
    postgres_port: int = Field(default=5433, ge=1, le=65535)
    postgres_db: str = 'lexaior_rag'
    postgres_user: str = 'lexaior'
    postgres_password: SecretStr
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    valkey_host: str = '127.0.0.1'
    valkey_port: int = Field(default=6380, ge=1, le=65535)
    valkey_password: SecretStr
    kafka_bootstrap_servers: str = '127.0.0.1:9093'
    kafka_topic_uploaded: str = 'doc.uploaded'
    kafka_topic_parsed: str = 'doc.parsed'
    kafka_topic_query: str = 'query.requested'
    # Folder pipeline topics (design doc); dead letters go to '<topic>_DLQ'.
    kafka_topic_standardization: str = 'STNDZ_TOPIC'
    kafka_topic_csv_parquet: str = 'CSV_PARQUET_TOPIC'
    kafka_topic_markdown: str = 'RAG_MARKDOWN_TOPIC'
    kafka_topic_image: str = 'RAG_IMAGE_TOPIC'
    kafka_topic_metaclassifier: str = 'METACLASSIFIER_TOPIC'
    storage_provider: str = 'seaweedfs'
    storage_endpoint_url: str = 'http://127.0.0.1:8333'
    storage_access_key: SecretStr
    storage_secret_key: SecretStr
    storage_bucket: str = 'rag-documents'
    storage_region: str = 'us-east-1'
    embedding_model: str = 'BAAI/bge-m3'
    embedding_model_revision: str | None = '5617a9f61b028005a4858fdac845db406aefb181'
    embedding_dimension: int = Field(default=1024, ge=1024, le=1024)
    embedding_device: str = Field(default='auto', pattern='^(auto|cpu|mps|cuda)$')
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    embedding_max_length: int = Field(default=1024, ge=128, le=8192)
    embedding_max_attempts: int = Field(default=3, ge=1, le=10)
    llm_base_url: str = 'http://127.0.0.1:11434/v1'
    llm_model: str = 'qwen3:4b'
    llm_api_key: SecretStr = SecretStr('ollama')
    reference_extraction_enabled: bool = False
    parse_timeout_seconds: int = Field(default=480, ge=5, le=600)
    parse_lease_seconds: int = Field(default=120, ge=30, le=600)
    parse_max_attempts: int = Field(default=3, ge=1, le=10)
    parse_max_pages: int = Field(default=500, ge=1, le=2000)
    parse_max_file_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    libreoffice_path: str = '/Applications/LibreOffice.app/Contents/MacOS/soffice'
    conversion_timeout_seconds: int = Field(default=180, ge=1, le=1800)
    csv_max_rows: int = Field(default=1_000_000, ge=1)
    csv_max_columns: int = Field(default=1000, ge=1)
    parser_threads: int = Field(default=4, ge=1, le=16)
    parser_force_ocr: bool = False
    standardization_group_id: str = 'lexaior-standardization-v1'
    indexing_group_id: str = 'lexaior-indexing-v1'
    chunk_target_tokens: int = Field(default=400, ge=64, le=2048)
    chunk_max_tokens: int = Field(default=512, ge=64, le=4096)
    index_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    index_lease_seconds: int = Field(default=180, ge=30, le=600)
    index_max_attempts: int = Field(default=3, ge=1, le=10)
    sweep_interval_seconds: int = Field(default=60, ge=5, le=3600)
    sweep_stuck_seconds: int = Field(default=900, ge=60, le=86400)
    sweep_batch_size: int = Field(default=50, ge=1, le=1000)
    # Folder pipeline fair queue (design doc UserAllocation): files running at once per stage,
    # and per user within a stage, counted across every worker from Postgres.
    standardization_max_workers: int = Field(default=2, ge=1, le=64)
    standardization_max_per_user: int = Field(default=1, ge=1, le=64)
    markdown_max_workers: int = Field(default=2, ge=1, le=64)
    markdown_max_per_user: int = Field(default=1, ge=1, le=64)
    image_max_workers: int = Field(default=1, ge=1, le=64)
    image_max_per_user: int = Field(default=1, ge=1, le=64)
    metaclassifier_max_workers: int = Field(default=1, ge=1, le=64)
    metaclassifier_max_per_user: int = Field(default=1, ge=1, le=64)
    parquet_max_workers: int = Field(default=2, ge=1, le=64)
    parquet_max_per_user: int = Field(default=1, ge=1, le=64)
    stage_lease_seconds: int = Field(default=180, ge=30, le=1800)
    stage_timeout_seconds: int = Field(default=1800, ge=30, le=7200)
    stage_max_attempts: int = Field(default=3, ge=1, le=10)
    stage_retry_base_seconds: int = Field(default=30, ge=0, le=3600)
    stage_poll_seconds: float = Field(default=30, ge=0.1, le=600)
    stage_stuck_seconds: int = Field(default=900, ge=60, le=86400)
    standardization_stuck_seconds: int = Field(default=1800, ge=60, le=86400)
    vision_enabled: bool = True
    vision_model: str = 'qwen3-vl:4b-instruct'
    vision_timeout_seconds: int = Field(default=180, ge=10, le=600)
    vision_max_images: int = Field(default=32, ge=1, le=256)
    vision_max_image_side: int = Field(default=1280, ge=256, le=2048)
    # Metaclassifier: a local instruct model reads the file's best passages and returns cited facts.
    metaclassifier_model: str = 'qwen3-vl:4b-instruct'
    metaclassifier_input_tokens: int = Field(default=6000, ge=500, le=100000)
    metaclassifier_context_tokens: int = Field(default=16384, ge=4096, le=262144)
    metaclassifier_timeout_seconds: int = Field(default=600, ge=30, le=3600)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        # Environment > mounted secrets > local development dotenv > defaults.
        return init_settings, env_settings, file_secret_settings, dotenv_settings

    @field_validator('postgres_password', 'valkey_password', 'storage_access_key', 'storage_secret_key')
    @classmethod
    def reject_placeholder_secrets(cls, value):
        if not value.get_secret_value().strip() or value.get_secret_value() == 'change-me':
            raise ValueError('Provide a real runtime secret; placeholders are not accepted.')
        return value

    @model_validator(mode='after')
    def chunk_budget(self):
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError('CHUNK_TARGET_TOKENS must not exceed CHUNK_MAX_TOKENS.')
        return self

    @field_validator('storage_provider')
    @classmethod
    def supported_storage(cls, value):
        if value != 'seaweedfs':
            raise ValueError('Only SeaweedFS is configured in RAG v1.')
        return value

    @property
    def database_url(self) -> URL:
        return URL.create('postgresql+asyncpg', username=self.postgres_user,
                          password=self.postgres_password.get_secret_value(),
                          host=self.postgres_host, port=self.postgres_port, database=self.postgres_db)


@lru_cache
def get_settings() -> Settings:
    default = Path(__file__).resolve().parents[2] / '.env'
    env_file = os.environ.get('RAG_ENV_FILE', str(default))
    return Settings(_env_file=env_file, _secrets_dir=os.environ.get('RAG_SECRETS_DIR'))
