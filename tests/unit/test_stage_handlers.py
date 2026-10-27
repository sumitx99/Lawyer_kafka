from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image
import pytest

from pyconsumers.core.rag_pipeline.execute_standardization import FolderStandardization
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedAsset, ParsedBlock, ParserOutput
from pyconsumers.core.rag_pipeline.userallocation.UserAllocation import Claim, StageError
from pyconsumers.helpers.secrets_control import Settings


def settings():
    return Settings(_env_file=None, postgres_password='t', valkey_password='t', storage_access_key='t', storage_secret_key='t')


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    async def get_bytes(self, key):
        return self.objects[key]

    async def put_bytes(self, key, data, content_type='application/octet-stream'):
        self.objects[key] = data


def png():
    buffer = BytesIO()
    Image.new('RGB', (8, 8), 'green').save(buffer, format='PNG')
    return buffer.getvalue()


def claim(storage, file_type, data, digest=None):
    doc, thread = uuid4(), uuid4()
    path = f'THREAD_{thread}/{file_type}/doc_{doc}/file.{file_type}'
    storage.objects[path] = data
    return Claim('standardization', 'w', {'id': doc, 'thread_id': thread, 'file_type': file_type, 's3_path': path,
                                          'sha256': digest or sha256(data).hexdigest()})


class Parser:
    def __init__(self, blocks, images=0):
        self.blocks, self.images = blocks, images

    async def parse(self, source, output):
        (output / 'assets').mkdir(parents=True)
        (output / 'document.md').write_text('# Order')
        assets = []
        for n in range(self.images):
            (output / f'assets/picture-{n}.png').write_bytes(png())
            assets.append(ParsedAsset(ref=f'#/pictures/{n}', path=f'assets/picture-{n}.png', provenance=[{'page_no': 1}]))
        return ParserOutput(parser_version='fixture', pages=[{'page_no': 1}, {'page_no': 2}], blocks=self.blocks,
                            assets=assets, docling_document={})


async def test_image_upload_skips_markdown_and_becomes_the_only_image():
    storage = MemoryStorage()
    c = claim(storage, 'png', png())
    result = await FolderStandardization(storage, settings(), parser=None)(c)
    assert result.next == {'markdown': 'INAPPLICABLE', 'image': 'WAITING'}
    manifest = ParserOutput.model_validate_json(storage.objects[result.summary['manifest_key']])
    assert [a.path for a in manifest.assets] == [c.row['s3_path']] and result.updates['s3_path_tomarkdown'] is None


async def test_pdf_next_stages_follow_its_content():
    storage = MemoryStorage()
    text = [ParsedBlock(ref='#/texts/0', label='text', text='Order of the court.', provenance=[{'page_no': 1}])]
    c = claim(storage, 'pdf', b'%PDF-1.7 fixture')
    result = await FolderStandardization(storage, settings(), parser=Parser(text, images=2))(c)
    assert result.next == {'markdown': 'WAITING', 'image': 'WAITING'} and result.updates['page_count'] == 2
    prefix = c.row['s3_path'].rsplit('/', 1)[0] + '/parsed'
    assert {f'{prefix}/document.md', f'{prefix}/document.json', f'{prefix}/assets/picture-1.png'} <= set(storage.objects)
    manifest = ParserOutput.model_validate_json(storage.objects[f'{prefix}/document.json'])
    assert manifest.assets[0].path == f'{prefix}/assets/picture-0.png'  # stored keys, not temp paths
    scanned_blank = await FolderStandardization(storage, settings(), parser=Parser([], images=0))(claim(storage, 'pdf', b'%PDF-x'))
    assert scanned_blank.next == {'markdown': 'INAPPLICABLE', 'image': 'INAPPLICABLE'}


@pytest.mark.parametrize('file_type,data,digest,message', [
    ('txt', b'plain\x00text', None, 'NUL'),
    ('pdf', b'%PDF-1.7', '0' * 64, 'does not match its upload checksum'),
    ('png', b'\x89PNG\r\n\x1a\nbroken', None, 'damaged or not a real image')])
async def test_unprocessable_files_fail_terminally(file_type, data, digest, message):
    storage = MemoryStorage()
    with pytest.raises(StageError, match=message) as error:
        await FolderStandardization(storage, settings(), parser=Parser([]))(claim(storage, file_type, data, digest))
    assert error.value.terminal


# ---- Metaclassifier: cited facts are kept only when printed -------------------------------------

from pyconsumers.core.rag_pipeline.execute_metaclassifier import DocumentFacts, Passage, iso_date, render, select_passages, verify

LEASE = [Passage('text', uuid4(), 0, 1, 'Made at Patna on 1 April 2026 between Asha Verma (Lessor) and Ravi Kumar (Lessee).',
                 'RENT AGREEMENT', []),
         Passage('text', uuid4(), 1, 1, '2.1 The Lessee shall pay a monthly rent of Rs. 15,000 by the 5th.', 'RENT AGREEMENT',
                 ['Operative', '2']),
         Passage('image', uuid4(), 0, 2, 'Image type: photo.\nA stamp reading STAMP DUTY Rs. 500 is present.', '', [])]


def facts(**changes):
    empty = {'value': '', 'evidence': []}
    return DocumentFacts.model_validate({
        'summary': 'A rent agreement.', 'document_type': 'rent_agreement', 'title': {'value': 'Rent Agreement', 'evidence': [1]},
        'parties': [{'name': 'Asha Verma', 'role': 'Lessor', 'evidence': [1]},
                    {'name': 'Sita Rao', 'role': 'Lessee', 'evidence': [1]}],  # invented: not printed anywhere
        'dates': [{'label': 'execution date', 'as_printed': '1 April 2026', 'iso': '2026-01-04', 'evidence': [1]}],
        'amounts': [{'label': 'rent', 'as_printed': 'Rs. 15,000', 'evidence': [2]},
                    {'label': 'stamp duty', 'as_printed': '₹500', 'evidence': [3]},
                    {'label': 'deposit', 'as_printed': 'Rs. 45,000', 'evidence': [2]}],  # invented amount
        'jurisdiction': {'value': 'Patna', 'evidence': [9]},  # cites a passage that does not exist
        'governing_law': empty, 'property': empty, 'court': empty, 'case_number': empty, 'citation': empty,
        'judges': [empty], 'statutes_cited': []} | changes)


def test_only_printed_and_cited_facts_survive():
    record, details, evidence, dropped = verify(facts(), LEASE)
    assert record['title'] == 'Rent Agreement'  # the heading counts as printed, case-insensitively
    assert record['parties'] == [{'name': 'Asha Verma', 'role': 'Lessor'}]
    assert details['dates'] == [{'label': 'execution date', 'as_printed': '1 April 2026', 'iso': '2026-04-01'}]  # printed date wins
    assert [a['as_printed'] for a in details['amounts']] == ['Rs. 15,000', '₹500']  # same number, other currency sign
    assert record['jurisdiction'] is None and record['judges'] == [] and record['decision_date'] is None
    assert dropped == ['parties[1]', 'amounts[2]', 'jurisdiction']  # empty placeholders are not "dropped"
    assert evidence['amounts[1]'] == [{'kind': 'image', 'id': str(LEASE[2].id), 'index': 0, 'page': 2}]


def test_decision_date_only_for_court_documents():
    order = facts(document_type='court_order', dates=[{'label': 'Date of order', 'as_printed': '1 April 2026', 'iso': None,
                                                        'evidence': [1]}])
    assert verify(order, LEASE)[0]['decision_date'].isoformat() == '2026-04-01'
    assert verify(facts(dates=order.dates), LEASE)[0]['decision_date'] is None


@pytest.mark.parametrize('printed,model,expected', [
    ('05/08/1980', '1980-05-08', '1980-08-05'),  # Indian day-first, whatever the model guessed
    ('१२ मार्च २०२६', None, '2026-03-12'), ('March 20, 2026', None, '2026-03-20'), ('5th day of April, 2026', None, '2026-04-05'),
    ('the fifth day of April 2026', '2026-04-05', '2026-04-05'), ('the fifth day of April 2026', '2026-05-05', None),
    ('2025', '2025-01-01', None), ('31/02/2026', None, None)])
def test_dates_are_read_from_the_printed_text(printed, model, expected):
    assert iso_date(printed, model) == expected


def test_passage_selection_keeps_opening_closing_and_signals_in_order():
    passages = [Passage('text', uuid4(), n, 1, f'Plain recital text number {n}.', '', [], 10) for n in range(20)]
    passages[10].content = 'Stamp duty of Rs. 500 paid on 01/04/2026 under Section 3 of the Act.'
    chosen = select_passages(passages, 60)
    # opening 0-2 (half), closing 19, then the passage with amounts/dates/statutes before plain passage 3
    assert [p.index for p in chosen] == [0, 1, 2, 3, 10, 19]
    assert render(chosen[:1]).startswith('[1] (page 1)\nPlain recital text number 0.')
    assert sum(p.tokens for p in chosen) <= 60


# ---- Phase G: text, Office conversion and CSV Parquet -----------------------------------------

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pyarrow as pa
import pyarrow.parquet as parquet

from pyconsumers.core.rag_pipeline.execute_parquet import CsvParquet


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-8-sig', 'utf-16', 'cp1252'])
async def test_text_blocks_keep_text_without_pages(encoding):
    text = 'Review in fourteen days.\r\n\r\n' + ('Résumé.' if encoding == 'cp1252' else 'नमस्ते न्यायालय।')
    storage = MemoryStorage()
    result = await FolderStandardization(storage, settings())(claim(storage, 'txt', text.encode(encoding)))
    manifest = ParserOutput.model_validate_json(storage.objects[result.summary['manifest_key']])
    assert manifest.parser == 'text' and len(manifest.blocks) == 2
    assert all(b.label == 'text' and not b.provenance for b in manifest.blocks)
    assert not manifest.pages and not manifest.assets and result.updates['page_count'] is None
    assert storage.objects[result.updates['s3_path_tomarkdown']].decode() == text
    assert result.next == {'markdown': 'WAITING', 'image': 'INAPPLICABLE'}


@pytest.mark.parametrize('data', [b'\xff\xfe\x00', b'\x81', 'bad\x00text'.encode('utf-16')])
async def test_invalid_text_is_terminal(data):
    storage = MemoryStorage()
    with pytest.raises(StageError) as error:
        await FolderStandardization(storage, settings())(claim(storage, 'txt', data))
    assert error.value.terminal


async def test_doc_conversion_uses_isolated_profile_and_pdf(monkeypatch):
    calls = []

    async def launch(*args, **kwargs):
        calls.append(args)
        assert args[1:3] == ('--headless', '--norestore')
        assert args[3].startswith('-env:UserInstallation=file://') and args[3].endswith('/lo-profile')
        Path(args[-1]).with_suffix('.pdf').write_bytes(b'%PDF')
        return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'', b'')))

    class PdfParser(Parser):
        async def parse(self, source, output):
            assert source.suffix == '.pdf' and source.read_bytes() == b'%PDF'
            return await super().parse(source, output)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', launch)
    storage = MemoryStorage()
    for _ in range(2):
        result = await FolderStandardization(storage, settings(), PdfParser([]))(claim(storage, 'doc', b'doc'))
        assert result.updates['page_count'] == 2
    assert calls[0][3] != calls[1][3]


async def test_doc_conversion_timeout_kills_process(monkeypatch):
    process = SimpleNamespace(returncode=None, communicate=AsyncMock(side_effect=asyncio.TimeoutError),
                              kill=Mock(), wait=AsyncMock())
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', AsyncMock(return_value=process))
    storage = MemoryStorage()
    with pytest.raises(asyncio.TimeoutError):
        await FolderStandardization(storage, settings(), Parser([]))(claim(storage, 'doc', b'doc'))
    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


async def test_doc_conversion_missing_pdf_is_terminal(monkeypatch):
    process = SimpleNamespace(returncode=1, communicate=AsyncMock(return_value=(b'', b'')))
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', AsyncMock(return_value=process))
    storage = MemoryStorage()
    with pytest.raises(StageError, match='no PDF') as error:
        await FolderStandardization(storage, settings(), Parser([]))(claim(storage, 'doc', b'doc'))
    assert error.value.terminal


@pytest.mark.parametrize('file_type,page_count', [('pptx', 2), ('xlsx', None)])
async def test_native_office_formats_use_parser(file_type, page_count):
    storage = MemoryStorage()
    result = await FolderStandardization(storage, settings(), Parser([ParsedBlock(ref='0', label='text', text='Review')]))(
        claim(storage, file_type, (Path(__file__).parents[1] / 'fixtures/sample.pptx').read_bytes()
              if file_type == 'pptx' else b'fixture'))
    assert result.updates['page_count'] == page_count
    assert result.next == {'markdown': 'WAITING', 'image': 'INAPPLICABLE'}


async def test_csv_writes_strings_and_summary():
    storage = MemoryStorage()
    c = claim(storage, 'csv', 'name;amount;code\nAsha;15000;001\nRavi;;002\n'.encode('utf-8-sig'))
    result = await CsvParquet(storage, settings())(c)
    summary = result.summary
    assert result.status == 'COMPLETED' and not result.updates
    assert summary['parquet_key'] == c.row['s3_path'].rsplit('/', 1)[0] + '/parquet/data.parquet'
    table = parquet.read_table(pa.BufferReader(storage.objects[summary['parquet_key']]))
    assert all(f.type == pa.string() for f in table.schema)
    assert table.to_pylist()[0] == {'name': 'Asha', 'amount': '15000', 'code': '001'}
    assert (summary['rows'], summary['columns'], summary['encoding'], summary['delimiter']) == (2, 3, 'utf-8-sig', ';')
    assert summary['column_names'] == ['name', 'amount', 'code']
    assert summary['null_counts'] == {'name': 0, 'amount': 1, 'code': 0}
    assert summary['sample_rows'] == table.to_pylist()
    assert summary['inferred_types']['amount'] == 'int64'
    assert parquet.ParquetFile(pa.BufferReader(storage.objects[summary['parquet_key']])).metadata.row_group(0).column(0).compression == 'ZSTD'


@pytest.mark.parametrize('data', [b'', b' \r\n', b'name,amount\n', b'name,amount'])
async def test_empty_csv_is_inapplicable(data):
    storage = MemoryStorage()
    c = claim(storage, 'csv', data)
    assert (await CsvParquet(storage, settings())(c)).status == 'INAPPLICABLE'
    assert list(storage.objects) == [c.row['s3_path']]


@pytest.mark.parametrize('data,changes,message', [
    (b'a,b,c\n1,2,3\n', {'csv_max_columns': 2}, 'column limit'),
    (b'a,b\n1,2\n3,4\n', {'csv_max_rows': 1}, 'row limit'),
    (b'a,b\n1,2,3\n', {}, 'malformed'),
    (b'a,a\n1,2\n', {}, 'unique')])
async def test_invalid_csv_is_terminal(data, changes, message):
    storage = MemoryStorage()
    c = claim(storage, 'csv', data)
    with pytest.raises(StageError, match=message) as error:
        await CsvParquet(storage, settings().model_copy(update=changes))(c)
    assert error.value.terminal and list(storage.objects) == [c.row['s3_path']]


async def test_csv_checksum_mismatch_is_terminal():
    storage = MemoryStorage()
    with pytest.raises(StageError, match='checksum') as error:
        await CsvParquet(storage, settings())(claim(storage, 'csv', b'a\n1\n', '0' * 64))
    assert error.value.terminal


@pytest.mark.parametrize('text', ['\nname,amount\nAsha,15000\n', 'name\nAsha\n'])
async def test_csv_blank_lines_and_single_column_preserve_string_schema(text):
    storage = MemoryStorage()
    result = await CsvParquet(storage, settings())(claim(storage, 'csv', text.encode()))
    table = parquet.read_table(pa.BufferReader(storage.objects[result.summary['parquet_key']]))
    assert result.summary['columns'] == table.num_columns
    assert result.summary['column_names'] == table.column_names
    assert all(f.type == pa.string() for f in table.schema)
    assert result.summary['delimiter'] == ',' and table['name'].to_pylist() == ['Asha']


@pytest.mark.parametrize('encoding', ['utf-16', 'cp1252'])
async def test_csv_legacy_encoding_and_preview_limit(encoding):
    storage = MemoryStorage()
    text = 'name,code\n' + ''.join(f'André,00{n}\n' for n in range(8))
    result = await CsvParquet(storage, settings())(claim(storage, 'csv', text.encode(encoding)))
    assert result.summary['encoding'] == encoding and result.summary['rows'] == 8
    assert len(result.summary['sample_rows']) == 5
    assert result.summary['sample_rows'][0] == {'name': 'André', 'code': '000'}


@pytest.mark.parametrize('mode', ['missing_chart', 'missing_text', 'conversion_failed', 'smartart'])
async def test_pptx_bad_native_output_falls_back_to_pdf(monkeypatch, mode):
    from pyconsumers.core.rag_pipeline.connectors.unstructured.utils.parser import ParseFailure
    import pyconsumers.core.rag_pipeline.execute_standardization as standardization

    source = Path(__file__).parents[1] / 'fixtures/sample_visual.pptx'
    calls = []

    class PresentationParser(Parser):
        async def parse(self, source, output, render_pages=None):
            calls.append((source.suffix, render_pages))
            if source.suffix == '.pptx' and mode == 'conversion_failed':
                raise ParseFailure('Native conversion failed.')
            result = await super().parse(source, output)
            if source.suffix == '.pdf':
                result.assets = [ParsedAsset(ref='slide', path='assets/picture-0.png', provenance=[{'page_no': 1}])]
                (output / 'assets/picture-0.png').write_bytes(png())
            return result

    async def convert(source, directory, settings):
        pdf = directory / 'source.pdf'
        pdf.write_bytes(b'%PDF')
        return pdf

    monkeypatch.setattr(standardization, 'convert_to_pdf', convert)
    if mode == 'smartart':
        monkeypatch.setattr(standardization, 'presentation_quality', lambda source, result: ('SmartArt requires rendered slide images.', [1]))
    storage = MemoryStorage()
    blocks = [] if mode == 'missing_text' else [ParsedBlock(ref='text', label='text', text='Title')]
    result = await FolderStandardization(storage, settings(), PresentationParser(blocks))(claim(storage, 'pptx', source.read_bytes()))
    assert calls == [('.pptx', None), ('.pdf', [1])]
    assert any('LibreOffice PDF fallback' in note for note in result.summary['warnings'])
    assert result.next['image'] == 'WAITING'
    manifest = ParserOutput.model_validate_json(storage.objects[result.summary['manifest_key']])
    assert manifest.assets[0].path.endswith('picture-0.png')


async def test_pptx_cancellation_does_not_start_fallback(monkeypatch):
    import pyconsumers.core.rag_pipeline.execute_standardization as standardization
    convert = AsyncMock()
    monkeypatch.setattr(standardization, 'convert_to_pdf', convert)
    parser = SimpleNamespace(parse=AsyncMock(side_effect=asyncio.CancelledError))
    storage = MemoryStorage()
    with pytest.raises(asyncio.CancelledError):
        await FolderStandardization(storage, settings(), parser)(claim(storage, 'pptx', b'fixture'))
    convert.assert_not_awaited()


def test_smartart_is_detected_even_when_native_text_exists(tmp_path):
    from pptx import Presentation
    from lxml import etree
    from pyconsumers.core.rag_pipeline.execute_standardization import presentation_quality
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = 'Fictional diagram'
    slide.element.append(etree.Element('{http://schemas.openxmlformats.org/drawingml/2006/diagram}relIds'))
    source = tmp_path / 'diagram.pptx'
    presentation.save(source)
    result = ParserOutput(parser_version='fixture', blocks=[ParsedBlock(ref='0', label='text', text='Fictional diagram')],
                          docling_document={})
    reason, pages = presentation_quality(source, result)
    assert 'SmartArt' in reason and pages == [1]


async def test_pptx_native_chart_image_keeps_native_output(monkeypatch):
    import pyconsumers.core.rag_pipeline.execute_standardization as standardization
    convert = AsyncMock()
    monkeypatch.setattr(standardization, 'convert_to_pdf', convert)
    storage = MemoryStorage()
    raw = (Path(__file__).parents[1] / 'fixtures/sample_visual.pptx').read_bytes()
    parser = Parser([ParsedBlock(ref='text', label='text', text='Fictional rent review')], images=1)
    result = await FolderStandardization(storage, settings(), parser)(claim(storage, 'pptx', raw))
    convert.assert_not_awaited()
    assert result.next == {'markdown': 'WAITING', 'image': 'WAITING'}
    assert not result.summary['warnings']


async def test_csv_null_markers_are_preserved_as_literal_strings():
    storage = MemoryStorage()
    values = ['NA', 'N/A', 'NULL', 'null', 'nan', '#N/A', 'NaN', 'None', ' ']
    raw = ('value,code\n' + ''.join(f'{value},{n}\n' for n, value in enumerate(values)) + ',empty\n"",quoted\n').encode()
    result = await CsvParquet(storage, settings())(claim(storage, 'csv', raw))
    table = parquet.read_table(pa.BufferReader(storage.objects[result.summary['parquet_key']]))
    assert table['value'].to_pylist() == values + [None, None]
    assert result.summary['null_counts'] == {'value': 2, 'code': 0}
    assert [row['value'] for row in result.summary['sample_rows']] == values[:5]


async def test_csv_preview_is_bounded_without_truncating_parquet():
    storage = MemoryStorage()
    long = 'न्याय' * 300
    raw = f'name,value\nAsha,{long}\nRavi,\n'.encode()
    result = await CsvParquet(storage, settings())(claim(storage, 'csv', raw))
    table = parquet.read_table(pa.BufferReader(storage.objects[result.summary['parquet_key']]))
    assert table['value'].to_pylist() == [long, None]
    assert result.summary['sample_rows'] == [{'name': 'Asha', 'value': long[:500]}, {'name': 'Ravi', 'value': None}]


@pytest.mark.parametrize('show', ['0', 'false'])
def test_pptx_hidden_slides_do_not_shift_rendered_pages_or_native_provenance(tmp_path, show):
    from pptx import Presentation
    from pyconsumers.core.rag_pipeline.execute_standardization import presentation_quality

    presentation = Presentation(Path(__file__).parents[1] / 'fixtures/sample_visual.pptx')
    hidden = presentation.slides.add_slide(presentation.slide_layouts[5])
    hidden.element.set('show', show)
    hidden.shapes.title.text = 'Hidden slide'
    ids = presentation.slides._sldIdLst
    ids.insert(0, ids[-1])
    source = tmp_path / 'hidden.pptx'
    presentation.save(source)
    result = ParserOutput(parser_version='fixture', blocks=[ParsedBlock(ref='text', label='text', text='Title')],
                          docling_document={})
    reason, pages = presentation_quality(source, result)
    assert 'chart images' in reason and pages == [1]
    result.assets = [ParsedAsset(ref='chart', path='chart.png', provenance=[{'page_no': 2}])]
    reason, pages = presentation_quality(source, result)
    assert reason is None and pages == [1]


def test_hidden_smartart_does_not_trigger_fallback(tmp_path):
    from pptx import Presentation
    from lxml import etree
    from pyconsumers.core.rag_pipeline.execute_standardization import presentation_quality

    presentation = Presentation()
    hidden = presentation.slides.add_slide(presentation.slide_layouts[5])
    hidden.element.set('show', '0')
    hidden.element.append(etree.Element('{http://schemas.openxmlformats.org/drawingml/2006/diagram}relIds'))
    presentation.slides.add_slide(presentation.slide_layouts[5])
    source = tmp_path / 'hidden-diagram.pptx'
    presentation.save(source)
    result = ParserOutput(parser_version='fixture', blocks=[], docling_document={})
    assert presentation_quality(source, result) == (None, [])
