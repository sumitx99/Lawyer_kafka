import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from PIL import Image
import pytest
from pydantic import ValidationError

from pyconsumers.core.rag_pipeline.connectors.unstructured.utils.vision import VisionEnricher
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedAsset, ParsedBlock, ParserOutput
from pyconsumers.core.rag_pipeline.schemas.visual import VisualObservation
from pyconsumers.helpers.llm_helper import LLMClient, LLMFailure
from pyconsumers.helpers.secrets_control import Settings
import httpx


def settings(**changes):
    return Settings(_env_file=None, postgres_password='test', valkey_password='test',
                    storage_access_key='test', storage_secret_key='test', **changes)


def observation(**changes):
    return dict(kind='chart', title='Quarterly sales', x_axis='Quarter', y_axis='Units sold',
                legend=[], visible_text=['Q1 10', 'Q2 20', 'Q3 30'],
                numeric_values=[{'label': 'Q1', 'value': '10', 'unit': 'units', 'evidence_text': 'Q1 10'}],
                observations=['Bars increase from Q1 to Q3.'], interpretation='Sales increase across quarters.',
                limitations=['No cause of the increase is shown.'], status='interpreted') | changes


def parser_output(tmp_path):
    Image.new('RGB', (100, 100), 'white').save(tmp_path / 'picture.png')
    (tmp_path / 'document.md').write_text('Source document')
    return ParserOutput(parser_version='fixture', blocks=[ParsedBlock(ref='#/pictures/0', label='picture')],
                        assets=[ParsedAsset(ref='#/pictures/0', path='picture.png', provenance=[{'page_no': 2}])],
                        warnings=['Images are extracted; charts, diagrams and photos are not semantically interpreted.'],
                        docling_document={})


async def test_visual_description_keeps_provenance_and_uses_project_cache(tmp_path):
    project = uuid4()
    store = {}
    async def read(key): return store.get(key)
    async def write(key, value, ttl): store[key] = value
    cache = SimpleNamespace(key=lambda *parts: ':'.join(map(str, parts)), get_json=read, set_json=write)
    client = SimpleNamespace(describe_image=AsyncMock(return_value=json.dumps(observation())))
    enricher = VisionEnricher(settings(), cache, client=client)
    result = await enricher.enrich(parser_output(tmp_path), tmp_path, project)
    assert result.assets[0].visual_description.model_derived
    block = result.blocks[-1]
    assert block.label == 'visual_description' and block.provenance == [{'page_no': 2}]
    assert block.data['origin'] == 'vision_model' and 'increase' in block.text
    assert 'Model-derived visual interpretation' in (tmp_path / 'document.md').read_text()
    assert not any('not semantically interpreted' in warning for warning in result.warnings)
    await enricher.enrich(parser_output(tmp_path), tmp_path, project)
    assert client.describe_image.await_count == 1
    await enricher.enrich(parser_output(tmp_path), tmp_path, uuid4())
    assert client.describe_image.await_count == 2


async def test_missing_vision_service_is_not_silently_skipped(tmp_path):
    client = SimpleNamespace(describe_image=AsyncMock(side_effect=LLMFailure('not available')))
    cache = SimpleNamespace(key=lambda *parts: 'test', get_json=AsyncMock(return_value=None))
    with pytest.raises(LLMFailure):
        await VisionEnricher(settings(), cache, client=client).enrich(parser_output(tmp_path), tmp_path, uuid4())


async def test_invalid_model_output_fails_retryably(tmp_path):
    client = SimpleNamespace(describe_image=AsyncMock(return_value='not valid JSON'))
    cache = SimpleNamespace(key=lambda *parts: 'test', get_json=AsyncMock(return_value=None))
    with pytest.raises(LLMFailure):
        await VisionEnricher(settings(), cache, client=client).enrich(parser_output(tmp_path), tmp_path, uuid4())


async def test_unclear_visual_has_explicit_limitation(tmp_path):
    result = parser_output(tmp_path)
    client = SimpleNamespace(describe_image=AsyncMock(return_value=json.dumps(observation(
        title=None, x_axis=None, y_axis=None, numeric_values=[], visible_text=[],
        interpretation='Three unlabeled bars increase in height; their subject is unknown.',
        limitations=['No title, units or axis labels are readable.'], status='insufficient_detail'))))
    cache = SimpleNamespace(key=lambda *parts: 'test', get_json=AsyncMock(return_value=None), set_json=AsyncMock())
    result = await VisionEnricher(settings(), cache, client).enrich(result, tmp_path, uuid4())
    assert any('Insufficient visual detail' in warning for warning in result.warnings)
    assert result.assets[0].visual_description.observation.numeric_values == []


def test_exact_numbers_need_transcribed_printed_evidence():
    with pytest.raises(ValidationError):
        VisualObservation.model_validate(observation(numeric_values=[{
            'label': 'Q1', 'value': '999', 'unit': 'units', 'evidence_text': 'Q1 10'}]))
    with pytest.raises(ValidationError):
        VisualObservation.model_validate(observation(numeric_values=[{
            'label': 'Q1', 'value': '999', 'unit': 'units', 'evidence_text': 'Q1 999'}]))


async def test_image_budget_is_explicit(tmp_path):
    result = parser_output(tmp_path)
    result.assets *= 2
    with pytest.raises(ValueError):
        await VisionEnricher(settings(vision_max_images=1), None).enrich(result, tmp_path, uuid4())


async def test_llm_transport_sends_image_and_schema(monkeypatch):
    response = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(observation())}}]}
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response)
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    raw = await LLMClient(settings()).describe_image(b'fake-image', 'inspect source', VisualObservation.model_json_schema())
    assert VisualObservation.model_validate_json(raw).kind == 'chart'
    assert requests[0]['response_format']['type'] == 'json_schema'
    assert requests[0]['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')


def test_observation_types_cover_tables_and_logos_and_prompt_keeps_seals_neutral():
    from pyconsumers.core.rag_pipeline.connectors.unstructured.utils.vision import PROMPT, PROMPT_VERSION
    for kind in ('table', 'logo'):
        assert VisualObservation.model_validate(observation(kind=kind)).kind == kind
    assert PROMPT_VERSION == 'visual-v2'
    assert 'never state that they are genuine' in PROMPT and 'stamp-duty' in PROMPT
