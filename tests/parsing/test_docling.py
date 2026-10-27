import os
from pathlib import Path

import pytest

from pyconsumers.core.rag_pipeline.connectors.unstructured.utils.parser import DoclingParser, ParseFailure
from pyconsumers.helpers.secrets_control import get_settings

pytestmark = pytest.mark.skipif(os.environ.get('RAG_RUN_PARSER') != '1',
                                reason='Set RAG_RUN_PARSER=1 for CPU conversion/model download tests.')
FIXTURES = Path(__file__).parents[1] / 'fixtures'


@pytest.mark.parametrize('name', ['normal.pdf', 'scanned.pdf', 'table_image.pdf', 'sample.docx', 'sample.pptx', 'sample.xlsx'])
async def test_real_docling_conversion(name, tmp_path):
    output = tmp_path / 'output'
    result = await DoclingParser(get_settings()).parse(FIXTURES / name, output)
    markdown = (output / 'document.md').read_text().lower()
    assert result.parser_version == '2.142.1'
    assert result.blocks
    if name in {'normal.pdf', 'scanned.pdf', 'sample.docx', 'sample.pptx'}:
        assert 'fourteen' in markdown
    if name == 'scanned.pdf':
        assert '12345' in markdown
    if name.endswith('.pdf'):
        assert result.pages
        assert any(block.provenance and block.provenance[0]['page_no'] >= 1 for block in result.blocks)
    if name == 'table_image.pdf':
        assert any(block.label == 'table' and block.data for block in result.blocks)
        assert result.assets
    if name == 'sample.docx':
        assert result.assets
        assert any('page numbers' in warning for warning in result.warnings)
        assert any(asset.ocr_text for asset in result.assets)
    for asset in result.assets:
        assert (output / asset.path).is_file()


async def test_corrupt_document_fails(tmp_path):
    with pytest.raises(ParseFailure):
        await DoclingParser(get_settings()).parse(FIXTURES / 'corrupt.pdf', tmp_path / 'output')


async def test_multi_page_table_rows_keep_pages(tmp_path):
    from pyconsumers.core.rag_pipeline.chunking.table_chunker import chunk_tables
    from pyconsumers.core.rag_pipeline.chunking.text_chunker import build_sections
    result = await DoclingParser(get_settings()).parse(FIXTURES / 'long_table.pdf', tmp_path / 'output')
    _, _, tables = build_sections(result.blocks)
    [table] = tables
    assert table.header == ['Date', 'Provider', 'Description', 'Amount (INR)']
    assert len(table.rows) == 55 and len(table.parts) == 2 and table.pages == [1, 2]
    assert table.rows[0].cells[2] == 'Treatment item 0' and table.rows[0].page == 1
    assert table.rows[-1].cells[2] == 'Treatment item 54' and table.rows[-1].page == 2
    chunks = chunk_tables(tables, lambda text: len(text.split()), 400, 512)
    assert all(chunk.content.count('| Date | Provider | Description | Amount (INR) |') == 1 for chunk in chunks)
    assert [n for c in chunks for n in range(c.source_metadata['row_start'], c.source_metadata['row_end'] + 1)] == list(range(1, 56))


async def test_lease_deed_chunks_follow_clauses(tmp_path):
    from pyconsumers.core.rag_pipeline.chunking.text_chunker import build_sections, chunk_sections
    result = await DoclingParser(get_settings()).parse(FIXTURES / 'lease_deed.pdf', tmp_path / 'output')
    sections, _, _ = build_sections(result.blocks)
    chunks = chunk_sections(sections, lambda text: len(text.split()), 80, 120)
    paths = [chunk.source_metadata['legal_path'] for chunk in chunks]
    # Docling merges lines into paragraphs and strips list numbers; every clause still gets its own chunk.
    assert paths == [[], ['Recital'], ['Recital'], ['Operative'], ['Operative', '1'], ['Operative', '2'], ['Operative', '3'],
                     ['Operative', '4'], ['Operative', '5'], ['Execution'], ['SCHEDULE A']]
    by_clause = {tuple(path): chunk for path, chunk in zip(paths, chunks)}
    rent = by_clause[('Operative', '2')]
    assert rent.content.startswith('2. Rent and deposit') and '2.1 The Lessee shall pay a monthly rent of Rs. 15,000' in rent.content
    assert '(b) after deducting the cost of damage' in rent.content and rent.page_start == 1
    assert by_clause[('Operative', '1')].source_metadata['defined_terms'] == ['Premises', 'Rent']
    assert by_clause[('Operative', '4')].page_start == 2 and '5.1' not in by_clause[('Operative', '4')].content


@pytest.mark.parametrize('name,count', [('sample.doc', None), ('sample.pptx', 2), ('sample.xlsx', None)])
async def test_real_office_standardization(name, count):
    from hashlib import sha256
    from types import SimpleNamespace
    from pyconsumers.core.rag_pipeline.execute_standardization import FolderStandardization
    from pyconsumers.core.rag_pipeline.schemas.parsed import ParserOutput

    class Storage:
        def __init__(self):
            self.objects = {}

        async def get_bytes(self, key):
            return self.objects[key]

        async def put_bytes(self, key, data, content_type):
            self.objects[key] = data

    storage = Storage()
    raw = (FIXTURES / name).read_bytes()
    key = f'fixture/{name}'
    storage.objects[key] = raw
    claim = SimpleNamespace(row={'s3_path': key, 'file_type': name.split('.')[-1], 'sha256': sha256(raw).hexdigest()})
    result = await FolderStandardization(storage, get_settings())(claim)
    markdown = storage.objects[result.updates['s3_path_tomarkdown']].decode().lower()
    manifest = ParserOutput.model_validate_json(storage.objects[result.summary['manifest_key']])
    if name.endswith('.xlsx'):
        assert 'asha verma' in markdown and '15000' in markdown
        assert any(b.label == 'table' for b in manifest.blocks)
    else:
        assert 'fourteen' in markdown
    if name.endswith('.doc'):
        assert result.updates['page_count'] >= 1
        assert manifest.pages and any(b.provenance for b in manifest.blocks)
    else:
        assert result.updates['page_count'] == count
    assert result.next['markdown'] == 'WAITING'


@pytest.mark.parametrize('hidden_slides', [False, True])
async def test_real_visual_pptx_falls_back_and_routes_slide_to_vision(tmp_path, hidden_slides):
    from hashlib import sha256
    from types import SimpleNamespace
    from pyconsumers.core.rag_pipeline.execute_standardization import FolderStandardization
    from pyconsumers.core.rag_pipeline.schemas.parsed import ParserOutput
    from PIL import Image
    from io import BytesIO

    raw = (FIXTURES / 'sample_visual.pptx').read_bytes()
    page = 1
    if hidden_slides:
        from pptx import Presentation
        presentation = Presentation(BytesIO(raw))
        intro = presentation.slides.add_slide(presentation.slide_layouts[5])
        intro.shapes.title.text = 'Visible introduction'
        ids = presentation.slides._sldIdLst
        ids.insert(0, ids[-1])
        for position in ('before', 'after'):
            hidden = presentation.slides.add_slide(presentation.slide_layouts[5])
            hidden.element.set('show', '0')
            hidden.shapes.title.text = 'Hidden slide must not appear in PDF'
            if position == 'before':
                ids.insert(0, ids[-1])
        buffer = BytesIO()
        presentation.save(buffer)
        raw = buffer.getvalue()
        page = 2
    objects = {'fixture/visual.pptx': raw}

    class Storage:
        async def get_bytes(self, key):
            return objects[key]

        async def put_bytes(self, key, data, content_type):
            objects[key] = data

    claim = SimpleNamespace(row={'s3_path': 'fixture/visual.pptx', 'file_type': 'pptx', 'sha256': sha256(raw).hexdigest()})
    result = await FolderStandardization(Storage(), get_settings())(claim)
    assert result.updates['page_count'] == page
    assert result.next == {'markdown': 'WAITING', 'image': 'WAITING'}
    assert any('LibreOffice PDF fallback' in note for note in result.summary['warnings'])
    manifest = ParserOutput.model_validate_json(objects[result.summary['manifest_key']])
    [slide] = [asset for asset in manifest.assets if asset.ref == f'#/rendered-slides/{page}']
    assert slide.provenance == [{'page_no': page}]
    image = Image.open(BytesIO(objects[slide.path]))
    assert image.width > 500 and image.height > 500
    image.save(tmp_path / 'rendered-slide.png')
    text = objects[result.updates['s3_path_tomarkdown']].decode().lower()
    assert 'fictional rent review' in text
    assert 'hidden slide' not in text
    if hidden_slides:
        assert 'visible introduction' in text
    assert any(block.ref == slide.ref and block.label == 'picture' for block in manifest.blocks)
