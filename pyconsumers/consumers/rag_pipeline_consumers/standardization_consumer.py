"""Consume doc.uploaded and invoke parsing (shared retry/DLQ/commit loop in consumer_helper)."""
from ...core.rag_pipeline.execute_standardization import Standardization
from ...helpers.consumer_helper import PipelineConsumer


class StandardizationConsumer(PipelineConsumer):
    event_type = 'doc.uploaded'
    done_states = frozenset({'parsed', 'duplicate'})
    failure_code = 'ParsingFailed'

    def __init__(self, db, storage, cache, settings, service=None):
        super().__init__(settings, service or Standardization(db, storage, cache, settings))

    @property
    def topic(self):
        return self.settings.kafka_topic_uploaded

    @property
    def group_id(self):
        return self.settings.standardization_group_id

    @property
    def max_poll_ms(self):
        s = self.settings
        vision = s.vision_timeout_seconds * s.vision_max_images if s.vision_enabled else 0
        return 1000 * (s.parse_max_attempts * (s.parse_timeout_seconds + 60 + vision) + s.parse_lease_seconds + 300)
