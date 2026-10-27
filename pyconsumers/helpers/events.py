"""Cross-service messages. Mirrored in the API repository (utilities/events.py).

RagEvent: per-document pipeline envelope.
ThreadMessage: folder pipeline message, exactly {"unstructured_thread_id": X} as in the design doc;
consumers read the folder's pending documents from Postgres.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RagEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal[1] = 1
    event_id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID = Field(default_factory=uuid4)
    event_type: Literal['doc.uploaded', 'doc.parsed', 'query.requested']
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    project_id: UUID
    doc_id: UUID | None = None
    query_id: UUID | None = None

    @model_validator(mode='after')
    def required_identity(self):
        if self.event_type.startswith('doc.') and (self.doc_id is None or self.query_id is not None):
            raise ValueError('Document events require doc_id and no query_id.')
        if self.event_type == 'query.requested' and (self.query_id is None or self.doc_id is not None):
            raise ValueError('Query events require query_id and no doc_id.')
        if self.occurred_at.tzinfo is None:
            raise ValueError('occurred_at must include a timezone.')
        return self


# Outbox event_type per stage, and the settings attribute holding that stage's topic.
THREAD_EVENTS = {
    'thread.standardization': 'kafka_topic_standardization',
    'thread.parquet': 'kafka_topic_csv_parquet',
    'thread.markdown': 'kafka_topic_markdown',
    'thread.image': 'kafka_topic_image',
    'thread.metaclassifier': 'kafka_topic_metaclassifier',
}


class ThreadMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    unstructured_thread_id: UUID
