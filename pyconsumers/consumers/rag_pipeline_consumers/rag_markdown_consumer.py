"""Consume doc.parsed and invoke indexing (shared retry/DLQ/commit loop in consumer_helper)."""
from ...core.rag_pipeline.execute_indexing import Indexing
from ...helpers.consumer_helper import PipelineConsumer


class IndexingConsumer(PipelineConsumer):
    event_type = 'doc.parsed'
    done_states = frozenset({'indexed', 'duplicate'})
    failure_code = 'IndexingFailed'

    def __init__(self, db, storage, cache, settings, service=None):
        super().__init__(settings, service or Indexing(db, storage, cache, settings))

    @property
    def topic(self):
        return self.settings.kafka_topic_parsed

    @property
    def group_id(self):
        return self.settings.indexing_group_id

    @property
    def max_poll_ms(self):
        s = self.settings
        return 1000 * (s.index_max_attempts * (s.index_timeout_seconds + 60) + s.index_lease_seconds + 300)
