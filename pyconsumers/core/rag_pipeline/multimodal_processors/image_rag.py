"""RAG_IMAGE stage: each image -> vision model -> type (table/graph/image) + description -> BGE-M3.

Logos are dropped. Descriptions are model-derived and marked as such; signatures, seals and stamps are
described as present only, never as genuine (see the vision prompt). Rows are replaced per file in one
transaction fenced by the stage lease, so re-running the stage never duplicates them.
"""
import json
from uuid import uuid5

from sqlalchemy import text

from ..connectors.unstructured.utils.vision import VisionEnricher, context_around
from ..indexing.embedder import Embedder
from ..schemas.parsed import ParserOutput
from ..userallocation.UserAllocation import StageError, StageResult, lock_claim

IMAGE_TYPE = {'table': 'table', 'chart': 'graph'}  # everything else is 'image'; logos are not stored


class ImageRag:
    def __init__(self, db, storage, cache, settings, model, vision=None):
        self.db, self.storage, self.settings = db, storage, settings
        self.vision = vision or VisionEnricher(settings, cache)
        self.embedder = Embedder(settings, model)

    async def __call__(self, claim):
        manifest_key = (claim.row['processing_summary'].get('standardization') or {}).get('manifest_key')
        if not manifest_key:
            raise StageError('No parsed manifest; retrigger standardization.', terminal=True)
        artifact = ParserOutput.model_validate_json(await self.storage.get_bytes(manifest_key))
        if len(artifact.assets) > self.settings.vision_max_images:
            raise StageError(f'{len(artifact.assets)} images exceed VISION_MAX_IMAGES={self.settings.vision_max_images}.',
                             terminal=True)
        positions = {block.ref: index for index, block in enumerate(artifact.blocks)}
        kept, counts = [], {'images': len(artifact.assets), 'logos_dropped': 0, 'insufficient_detail': 0}
        for index, asset in enumerate(artifact.assets):
            data = await self.storage.get_bytes(asset.path)
            context = context_around(artifact.blocks, positions.get(asset.ref, 0))
            description = await self.vision.describe(data, context, claim.thread_id)
            observation = description.observation
            if observation.kind == 'logo':
                counts['logos_dropped'] += 1
                continue
            counts['insufficient_detail'] += observation.status == 'insufficient_detail'
            page = next((item['page_no'] for item in asset.provenance if isinstance(item.get('page_no'), int)), None)
            kept.append({'index': index, 'asset': asset, 'page': page, 'type': IMAGE_TYPE.get(observation.kind, 'image'),
                         'text': VisionEnricher.to_text(observation), 'description': description})
        vectors = await self.embedder.embed([item['text'] for item in kept])
        async with self.db.transaction() as session:
            await lock_claim(session, claim)
            await session.execute(text('DELETE FROM image_rag_vector_table WHERE doc_id=:doc'), {'doc': claim.doc_id})
            if kept:
                await session.execute(text("""INSERT INTO image_rag_vector_table (id,doc_id,thread_id,image_index,s3_path,
                    page_no,image_type,description,metadata,embedding,embedding_model)
                    VALUES (:id,:doc,:thread,:index,:path,:page,:type,:description,CAST(:meta AS jsonb),
                    CAST(:embedding AS vector),:model)"""),
                    [{'id': uuid5(claim.doc_id, f'image:{item["index"]}'), 'doc': claim.doc_id, 'thread': claim.thread_id,
                      'index': item['index'], 'path': item['asset'].path, 'page': item['page'], 'type': item['type'],
                      'description': item['text'], 'model': self.settings.embedding_model,
                      'meta': json.dumps({'ref': item['asset'].ref, 'model_derived': True,
                                          **item['description'].model_dump(mode='json')}),
                      'embedding': '[' + ','.join(repr(float(v)) for v in vector) + ']'}
                     for item, vector in zip(kept, vectors, strict=True)])
        counts |= {'stored': len(kept), 'vision_model': self.settings.vision_model}
        return StageResult(status='COMPLETED' if artifact.assets else 'INAPPLICABLE', summary=counts)
