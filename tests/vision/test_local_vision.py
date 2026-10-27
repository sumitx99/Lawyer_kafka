import os
from pathlib import Path
from uuid import uuid4
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from pyconsumers.core.rag_pipeline.connectors.unstructured.utils.vision import VisionEnricher
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedAsset, ParsedBlock, ParserOutput
from pyconsumers.helpers.secrets_control import get_settings

pytestmark = pytest.mark.skipif(os.environ.get('RAG_RUN_VISION') != '1',
                                reason='Requires local Ollama vision model.')
FIXTURES = Path(__file__).parents[1] / 'fixtures'


@pytest.mark.parametrize('filename', ['labeled_chart.png', 'chart.png'])
async def test_real_chart_meaning(tmp_path, filename):
    (tmp_path / 'picture.png').write_bytes((FIXTURES / filename).read_bytes())
    (tmp_path / 'document.md').write_text('Fictional chart.')
    result = ParserOutput(parser_version='fixture', blocks=[ParsedBlock(ref='#/pictures/0', label='picture')],
                          assets=[ParsedAsset(ref='#/pictures/0', path='picture.png', provenance=[{'page_no': 1}])],
                          docling_document={})
    cache = SimpleNamespace(key=lambda *parts: 'test', get_json=AsyncMock(return_value=None), set_json=AsyncMock())
    result = await VisionEnricher(get_settings(), cache).enrich(result, tmp_path, uuid4())
    observation = result.assets[0].visual_description.observation
    assert observation.kind == 'chart'
    text = result.blocks[-1].text.lower()
    assert any(word in text for word in ['increase', 'increasing', 'rise', 'rising', 'upward', 'grow'])
    if filename == 'labeled_chart.png':
        assert 'sales' in (observation.title or '').lower()
        assert 'q1' in text and 'q3' in text
    else:
        assert observation.title is None
        assert observation.x_axis is None and observation.y_axis is None
        assert observation.numeric_values == []
        assert observation.limitations
    assert result.blocks[-1].provenance == [{'page_no': 1}]
