from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from pyconsumers.consumers.rag_pipeline_consumers.rag_markdown_consumer import IndexingConsumer
from pyconsumers.core.rag_pipeline.chunking.text_chunker import build_sections, chunk_sections
from pyconsumers.core.rag_pipeline.indexing.embedder import Embedder, EmbeddingFailure
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedBlock
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.secrets_control import Settings


def settings(**changes):
    return Settings(_env_file=None, postgres_password='test', valkey_password='test',
                    storage_access_key='test', storage_secret_key='test', **changes)


def words(text):
    return len(text.split())


def block(label, text='', page=None, level=None, ref=None):
    provenance = [{'page_no': page, 'bbox': {'l': 1, 't': 2, 'r': 3, 'b': 0}}] if page else []
    return ParsedBlock(ref=ref or f'#/{label}/{uuid4().hex[:6]}', label=label, text=text, level=level, provenance=provenance)


def test_sections_follow_heading_hierarchy_and_skip_non_text():
    sections, skipped, _ = build_sections([
        block('title', 'Order of the Court'),
        block('text', 'Preamble text.', page=1),
        block('section_header', 'Facts', level=1),
        block('section_header', 'Background', level=2),
        block('text', 'The parties signed on 1 May.', page=2),
        block('page_footer', 'Page 2 of 9', page=2),
        block('table', '| a | b |', page=2),
        block('section_header', 'Findings', level=1),
        block('section_header', 'Empty subsection', level=2),
        block('section_header', 'Held', level=2),
        block('visual_description', 'Image type: photo.', page=3),
        block('picture_ocr', 'STAMP DUTY PAID', page=3),
    ])
    assert [s.heading_path for s in sections] == [['Order of the Court'], ['Order of the Court', 'Facts', 'Background'],
                                                  ['Order of the Court', 'Findings', 'Held']]
    assert skipped == {'page_footer': 1, 'table': 1}
    assert 'Page 2 of 9' not in sections[1].content
    held = sections[2].content
    assert held.startswith('[Model-derived image interpretation') and '[Text read from an image by OCR]' in held


def test_chunks_carry_pages_locators_and_respect_limits():
    long_text = ' '.join(f'Sentence {n} states a fact about the claim.' for n in range(40))
    sections, _, _ = build_sections([block('section_header', 'Facts', level=1),
                                  block('text', 'Short opening.', page=1), block('text', long_text, page=2),
                                  block('visual_description', 'Bar chart of payments.', page=3)])
    chunks = chunk_sections(sections, words, target=60, limit=80)
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    joined = ' '.join(c.content for c in chunks)
    assert words(joined) == words('Short opening. ' + long_text) + words('[Model-derived image interpretation; '
                                                                        'verify against the source image] Bar chart of payments.')
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 3
    assert all(c.embedding_text.startswith('Facts\n\n') for c in chunks)
    assert chunks[-1].source_metadata['model_derived'] is True
    assert chunks[0].source_metadata['locator'] == {'section_ordinal': 0, 'chunk_in_section': 0}
    assert chunks[0].source_metadata['boxes'][0]['page_no'] == 1


def test_docx_chunks_never_invent_pages():
    sections, _, _ = build_sections([block('section_header', 'Clause 1', level=1), block('text', 'Rent is due monthly.')])
    [chunk] = chunk_sections(sections, words, target=50, limit=60)
    assert (chunk.page_start, chunk.page_end) == (None, None)
    assert chunk.source_metadata['block_refs'] and chunk.source_metadata['boxes'] == []


def test_single_overlong_word_run_is_split_with_progress():
    sections, _, _ = build_sections([block('text', ' '.join(['word'] * 250), page=1)])
    chunks = chunk_sections(sections, words, target=40, limit=50)
    assert sum(c.token_count for c in chunks) == 250 and all(c.token_count <= 50 for c in chunks)


class FakeModel:
    def __init__(self, failures=0, dimension=1024):
        self.failures, self.dimension, self.batches = failures, dimension, []

    async def embed_async(self, texts):
        self.batches.append(len(texts))
        if self.failures:
            self.failures -= 1
            raise RuntimeError('transient')
        return [[0.1] * self.dimension for _ in texts]


async def test_embedder_batches_and_retries(monkeypatch):
    monkeypatch.setattr('asyncio.sleep', AsyncMock())
    model = FakeModel(failures=1)
    vectors = await Embedder(settings(embedding_batch_size=4), model).embed([f't{n}' for n in range(10)])
    assert len(vectors) == 10 and model.batches == [4, 4, 4, 2]


async def test_embedder_fails_after_attempts_and_rejects_bad_vectors(monkeypatch):
    monkeypatch.setattr('asyncio.sleep', AsyncMock())
    with pytest.raises(EmbeddingFailure):
        await Embedder(settings(embedding_max_attempts=2), FakeModel(failures=5)).embed(['a'])
    with pytest.raises(EmbeddingFailure):
        await Embedder(settings(embedding_max_attempts=1), FakeModel(dimension=768)).embed(['a'])


def test_chunk_target_cannot_exceed_limit():
    with pytest.raises(ValueError):
        settings(chunk_target_tokens=600, chunk_max_tokens=512)


def record_for(event):
    return SimpleNamespace(topic='doc.parsed', partition=0, offset=3, value=event.model_dump_json().encode(), key=b'doc')


async def test_indexing_consumer_commits_after_index_and_dead_letters_failures():
    event = RagEvent(event_type='doc.parsed', project_id=uuid4(), doc_id=uuid4())
    service = AsyncMock()
    service.execute.return_value = 'indexed'
    consumer, producer = AsyncMock(), SimpleNamespace(client=AsyncMock())
    await IndexingConsumer(None, None, None, settings(), service=service).process_record(record_for(event), consumer, producer)
    consumer.commit.assert_awaited_once()
    producer.client.send_and_wait.assert_not_awaited()

    service.execute.return_value = 'failed'
    consumer, producer = AsyncMock(), SimpleNamespace(client=AsyncMock())
    await IndexingConsumer(None, None, None, settings(), service=service).process_record(record_for(event), consumer, producer)
    assert producer.client.send_and_wait.await_args.args[0] == 'doc.parsed.dlq'
    service.repo.acknowledge_failed.assert_awaited_once()
    consumer.commit.assert_awaited_once()


async def test_indexing_consumer_rejects_wrong_event_type():
    event = RagEvent(event_type='doc.uploaded', project_id=uuid4(), doc_id=uuid4())
    service, consumer, producer = AsyncMock(), AsyncMock(), SimpleNamespace(client=AsyncMock())
    await IndexingConsumer(None, None, None, settings(), service=service).process_record(record_for(event), consumer, producer)
    service.execute.assert_not_awaited()
    consumer.commit.assert_awaited_once()


def cell(text, header=False, section=False):
    return {'text': text, 'column_header': header, 'row_header': False, 'row_section': section}


def table(ref, page, header, rows, sections=()):
    grid = ([[cell(value, header=True) for value in header]] if header else []) + [
        [cell(value, section=index in sections) for value in row] for index, row in enumerate(rows)]
    return ParsedBlock(ref=ref, label='table', text='| table |', data={'grid': grid},
                       provenance=[{'page_no': page, 'bbox': {'l': 1, 't': 2, 'r': 3, 'b': 0}}] if page else [])


EXPENSES = ['Date', 'Provider', 'Amount']


def test_multi_page_table_is_merged_with_row_pages_and_repeated_headers():
    from pyconsumers.core.rag_pipeline.chunking.table_chunker import chunk_tables
    page_one = [[f'0{n}/01', f'Hospital {n}', f'{n}000'] for n in range(1, 8)]
    page_two = [[f'1{n}/02', f'Clinic {n}', f'{n}500'] for n in range(1, 6)]
    sections, skipped, tables = build_sections([
        block('section_header', 'Medical expenses', level=1),
        table('#/tables/0', 1, EXPENSES, page_one),
        block('page_footer', 'Page 1', page=1), block('page_header', 'Case 42/2026', page=2),
        table('#/tables/1', 2, EXPENSES, page_two),  # repeated header on the next page
        block('text', 'Total claimed is shown above.', page=2),
    ], captions={'#/tables/0': 'Table 1: Statement of medical expenses'})
    assert len(tables) == 1 and [p.ref for p in tables[0].parts] == ['#/tables/0', '#/tables/1']
    assert len(tables[0].rows) == 12 and tables[0].rows[7].page == 2 and tables[0].rows[0].page == 1
    chunks = chunk_tables(tables, words, target=40, limit=60)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.chunk_type == 'table' and chunk.table_id == '#/tables/0'
        assert '| Date | Provider | Amount |' in chunk.content and chunk.content.startswith('Table 1: Statement')
        assert chunk.embedding_text.startswith('Medical expenses\nTable 1: Statement of medical expenses\nTable columns: Date, Provider, Amount')
        meta = chunk.source_metadata
        assert all(chunk.page_start <= page <= chunk.page_end for page in meta['row_pages'].values())
    assert 'Date: 01/01; Provider: Hospital 1; Amount: 1000' in chunks[0].embedding_text
    numbers = [n for c in chunks for n in range(c.source_metadata['row_start'], c.source_metadata['row_end'] + 1)]
    assert numbers == list(range(1, 13))
    assert chunks[-1].page_end == 2 and chunks[-1].source_metadata['multi_page'] is True
    assert '| Date | Provider | Amount |' in sections[0].content


def test_tables_are_not_merged_across_headings_or_column_changes():
    _, _, tables = build_sections([
        table('#/tables/0', 1, EXPENSES, [['01/01', 'Hospital', '100']]),
        table('#/tables/1', 2, ['Witness', 'Statement'], [['A', 'Saw the collision']]),  # different columns
        block('section_header', 'Annexure B', level=1),
        table('#/tables/2', 3, ['Witness', 'Statement'], [['B', 'Heard the impact']]),  # heading in between
        table('#/tables/3', 5, ['Witness', 'Statement'], [['C', 'Arrived later']]),  # skips a page
    ])
    assert [t.table_id for t in tables] == ['#/tables/0', '#/tables/1', '#/tables/2', '#/tables/3']
    assert tables[2].heading_path == ['Annexure B']


def test_row_sections_headerless_tables_and_document_order():
    from pyconsumers.core.rag_pipeline.chunking.table_chunker import chunk_tables
    sections, _, tables = build_sections([
        block('section_header', 'Damages', level=1),
        block('text', 'The claimant lists the following heads.', page=1),
        table('#/tables/0', 1, [], [['Hospital charges', '', ''], ['Surgery', '01/02', '90000'], ['Medicines', '03/02', '4000']],
              sections=(0,)),
        block('text', 'All amounts are in rupees.', page=1),
    ])
    [chunk] = chunk_tables(tables, words, target=200, limit=300)
    assert chunk.source_metadata['header'] == [] and '| Surgery | 01/02 | 90000 |' in chunk.content
    assert '[Hospital charges] Column 1: Surgery; Column 2: 01/02; Column 3: 90000' in chunk.embedding_text
    text_chunks = chunk_sections(sections, words, target=100, limit=200)  # budget fits both texts; the table must still split them
    ordered = sorted(text_chunks + [chunk], key=lambda c: (c.section_ordinal, c.position))
    assert [c.chunk_type for c in ordered] == ['text', 'table', 'text']


def test_docx_tables_have_no_pages():
    from pyconsumers.core.rag_pipeline.chunking.table_chunker import chunk_tables
    _, _, tables = build_sections([table('#/tables/0', None, ['Clause', 'Rent'], [['5.1', '25000']])])
    [chunk] = chunk_tables(tables, words, target=50, limit=60)
    assert (chunk.page_start, chunk.page_end) == (None, None) and chunk.source_metadata['row_pages'] == {}


def test_text_search_config_per_document_language():
    from pyconsumers.core.rag_pipeline.execute_indexing import text_search_config
    assert text_search_config(['The tenant shall pay rent monthly.']) == ('english', 'en')
    assert text_search_config(['किरायेदार हर महीने किराया देगा।']) == ('simple', None)
    assert text_search_config(['Clause 5: किराया देय है', 'payment']) == ('simple', None)  # mixed scripts: no English stemming
    assert text_search_config([]) == ('simple', None)


def test_legal_markers_and_nested_subclauses():
    from pyconsumers.core.rag_pipeline.chunking.text_chunker import Unit, assign_legal_paths, legal_marker
    assert legal_marker('Clause 5. Rent') == ('clause', 'Clause 5', 2)
    assert legal_marker('5.2 The tenant shall pay') == ('numbered', '5.2', 3)
    assert legal_marker('5.2.1 Late fee') == ('numbered', '5.2.1', 4)
    assert legal_marker('7) Notices') == ('numbered', '7', 2)
    assert legal_marker('WHEREAS the Lessor owns the flat') == ('recital', 'Recital', 2)
    assert legal_marker('NOW THIS DEED WITNESSETH as follows') == ('operative', 'Operative', 1)
    assert legal_marker('SCHEDULE A - Property') == ('schedule', 'SCHEDULE A', 1)
    assert legal_marker('IN WITNESS WHEREOF the parties have signed') == ('execution', 'Execution', 1)
    assert legal_marker('धारा 138 के अंतर्गत') == ('clause', 'धारा 138', 2)
    assert legal_marker('Section ५ applies') == ('clause', 'Section ५', 2)
    for plain in ('25000 rupees are payable', '2 months notice', 'The Section applies', '2024 was a year'):
        assert legal_marker(plain) is None
    units = [Unit(text, f'#/texts/{n}', 'text', []) for n, text in enumerate(
        ['5. Rent', '5.1 Payment is due monthly.', '(a) by cheque;', '(i) drawn on a bank;', '(ii) signed;', '(b) by transfer.',
         'It is payable in advance.', '(h) eighth', '(i) ninth', '6. Deposit'])]
    assign_legal_paths(units)
    assert [u.legal_path for u in units] == [
        ['5'], ['5', '5.1'], ['5', '5.1', '(a)'], ['5', '5.1', '(a)', '(i)'], ['5', '5.1', '(a)', '(ii)'], ['5', '5.1', '(b)'],
        ['5', '5.1', '(b)'],  # a continuation paragraph stays in its clause
        ['5', '5.1', '(h)'], ['5', '5.1', '(i)'],  # (i) after (h) is the next letter, not a roman sub-item
        ['6']]


def test_chunks_never_span_two_clauses_and_carry_the_clause_path():
    long_clause = ' '.join(f'The tenant shall keep item {n} in good repair at Rs. 500 per month.' for n in range(12))
    sections, _, _ = build_sections([
        block('section_header', 'Lease deed', level=1),
        block('text', 'WHEREAS the Lessor owns flat No. 4.', page=1),
        block('text', 'AND WHEREAS the Lessee wishes to rent it.', page=1),
        block('text', '1. "Premises" means flat No. 4 at Patna.', page=1),
        block('text', '2. Rent', page=1),
        block('text', '2.1 ' + long_clause, page=2),
        block('text', '2.2 Rent is due on the 5th.', page=2),
        block('text', '3. Term of eleven months.', page=3)])
    chunks = chunk_sections(sections, words, target=40, limit=60)
    paths = [c.source_metadata['legal_path'] for c in chunks]
    assert paths[:3] == [['Recital'], ['Recital'], ['1']]  # two recitals: two chunks, no clause shares a chunk
    assert chunks[2].source_metadata['defined_terms'] == ['Premises']
    rent = [c for c in chunks if c.source_metadata['legal_path'][:1] == ['2']]
    assert len(rent) >= 3 and rent[0].content.startswith('2. Rent')
    split_piece = rent[1]  # a piece of the long clause 2.1 without its number still knows where it is
    assert split_piece.source_metadata['legal_path'] == ['2', '2.1']
    assert split_piece.embedding_text.startswith('Lease deed > 2 > 2.1\n\n')
    assert not any('Rs.' == c.content.split()[-1] for c in chunks)  # "Rs. 500" is never cut after "Rs."
    assert chunks[-1].content == '3. Term of eleven months.' and all(c.token_count <= 60 for c in chunks)
    assert {c.source_metadata['chunker_version'] for c in chunks} == {'legal-v1'}


def test_sentence_split_keeps_legal_abbreviations():
    from pyconsumers.core.rag_pipeline.chunking.text_chunker import split_sentences
    text = 'Pay Rs. 5,000 to Smt. Devi u/s. 34 of the Act. See S. K. Rao vs. State, i.e. the leading case. Done.'
    assert split_sentences(text) == ['Pay Rs. 5,000 to Smt. Devi u/s. 34 of the Act.',
                                     'See S. K. Rao vs. State, i.e. the leading case.', 'Done.']


def test_merged_paragraphs_split_only_at_safe_inline_markers():
    from pyconsumers.core.rag_pipeline.chunking.text_chunker import split_inline_units
    merged = ('Made at Patna. WHEREAS the Lessor owns flat No. 4. AND WHEREAS the Lessee agrees: (a) to pay; (b) to repair. '
              '4. Maintenance 4.1 The Lessee shall keep it clean. IN WITNESS WHEREOF signed. SCHEDULE A Flat No. 4.')
    assert split_inline_units(merged) == [
        'Made at Patna.', 'WHEREAS the Lessor owns flat No. 4.', 'AND WHEREAS the Lessee agrees:', '(a) to pay;', '(b) to repair.',
        '4. Maintenance', '4.1 The Lessee shall keep it clean.', 'IN WITNESS WHEREOF signed.', 'SCHEDULE A Flat No. 4.']
    for whole in ('Pay Rs. 4.5 Lakh by cheque.', 'Liable under Section 138. Section 141 also applies.',
                  'The court whereas held otherwise.', 'Paid on 1. 5. 2026 as agreed.', 'See flat No. 4. Then leave.',
                  'Total 1,50,000.00 Rupees.'):
        assert split_inline_units(whole) == [whole]
