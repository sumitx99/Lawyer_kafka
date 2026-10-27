"""Produce table row chunks with repeated headers and page provenance.

Rows are packed into chunks up to a token target and every chunk repeats the column header,
so a chunk from the middle of a long table still reads correctly. The embedding text uses
"Column: value" pairs per row; the stored content stays a Markdown table for display.
A table Docling split across pages is merged back when the next part sits on the following
page with the same column count and either repeats the header or has none. Each row keeps
the page of the part it came from.
"""
from dataclasses import dataclass, field

from . import IndexChunk

TABLE_CHUNKER_VERSION = 'table-rows-v1'


@dataclass
class TableRow:
    number: int  # 1-based data row number within the logical table
    cells: list[str]
    page: int | None
    part_ref: str
    group: str | None = None  # nearest preceding row_section label, e.g. "Hospital charges"


@dataclass
class TablePart:
    ref: str
    page: int | None
    bbox: dict | None
    columns: int
    header: list[str]
    rows: list[list[str]]
    groups: list[str | None]


@dataclass
class LogicalTable:
    table_id: str
    section_ordinal: int
    heading_path: list[str]
    position: int
    columns: int
    header: list[str]
    caption: str = ''
    parts: list[TablePart] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)

    @property
    def pages(self):
        return sorted({row.page for row in self.rows if row.page})

    def markdown(self, rows=None):
        rows = self.rows if rows is None else rows
        lines = []
        if self.header:
            lines += ['| ' + ' | '.join(self.header) + ' |', '|' + '---|' * self.columns]
        lines += ['| ' + ' | '.join(row.cells) + ' |' for row in rows]
        text = '\n'.join(lines)
        return f'{self.caption}\n\n{text}' if self.caption else text


def clean(value):
    return ' '.join(str(value or '').split()).replace('|', '/')


def read_part(block) -> TablePart | None:
    """Turn a Docling table block into header names and data rows (spanned cells repeat per column)."""
    grid = (block.data or {}).get('grid') or []
    if not grid:
        return None
    columns = max(len(row) for row in grid)
    header_rows, index = [], 0
    while index < len(grid) and any(cell.get('text') for cell in grid[index]) and all(
            cell.get('column_header') or not clean(cell.get('text')) for cell in grid[index]):
        header_rows.append([clean(cell.get('text')) for cell in grid[index]])
        index += 1
    header = []
    for column in range(columns):
        names = [row[column] for row in header_rows if column < len(row) and row[column]]
        header.append(' / '.join(dict.fromkeys(names)) or f'Column {column + 1}')
    if not header_rows:
        header = []
    rows, groups, group = [], [], None
    for row in grid[index:]:
        values = [clean(cell.get('text')) for cell in row] + [''] * (columns - len(row))
        if not any(values):
            continue
        if any(cell.get('row_section') for cell in row):
            group = ' '.join(dict.fromkeys(value for value in values if value))
            continue
        rows.append(values)
        groups.append(group)
    page = next((item['page_no'] for item in block.provenance if isinstance(item.get('page_no'), int)), None)
    bbox = next((item.get('bbox') for item in block.provenance if item.get('bbox')), None)
    return TablePart(block.ref, page, bbox, columns, header, rows, groups)


def continues(table: LogicalTable, part: TablePart) -> bool:
    last_page = table.parts[-1].page
    return (part.columns == table.columns and last_page is not None and part.page == last_page + 1
            and (not part.header or part.header == table.header))


def add_part(table: LogicalTable, part: TablePart):
    table.parts.append(part)
    for cells, group in zip(part.rows, part.groups):
        table.rows.append(TableRow(len(table.rows) + 1, cells, part.page, part.ref, group))


def chunk_tables(tables: list[LogicalTable], count, target: int, limit: int) -> list:
    chunks = []
    for table in tables:
        if not table.rows:
            continue
        heading = ' > '.join(table.heading_path)
        fixed = count(table.markdown([]))
        groups, current, used = [], [], fixed
        for row in table.rows:
            size = count('| ' + ' | '.join(row.cells) + ' |')
            if current and used + size > target:
                groups.append(current)
                current, used = [], fixed
            current.append(row)
            used += size
        if current:
            groups.append(current)
        for position, rows in enumerate(groups):
            content = table.markdown(rows)
            pairs = []
            for row in rows:
                labels = table.header or [f'Column {n + 1}' for n in range(table.columns)]
                pair = '; '.join(f'{name}: {value}' for name, value in zip(labels, row.cells) if value)
                pairs.append(f'[{row.group}] {pair}' if row.group else pair)
            context = '\n'.join(item for item in [heading, table.caption,
                                                  'Table columns: ' + ', '.join(table.header) if table.header else ''] if item)
            pages = sorted({row.page for row in rows if row.page})
            refs = list(dict.fromkeys(row.part_ref for row in rows))
            token_count = count(content)
            chunks.append(IndexChunk(
                ordinal=0, section_ordinal=table.section_ordinal, heading_path=table.heading_path,
                content=content, embedding_text=f'{context}\n\n' + '\n'.join(pairs) if context else '\n'.join(pairs),
                token_count=token_count, page_start=pages[0] if pages else None, page_end=pages[-1] if pages else None,
                position=table.position, chunk_type='table', table_id=table.table_id,
                source_metadata={
                    'chunker_version': TABLE_CHUNKER_VERSION, 'table_id': table.table_id,
                    'block_refs': refs, 'labels': ['table'], 'header': table.header, 'caption': table.caption,
                    'row_start': rows[0].number, 'row_end': rows[-1].number, 'table_row_count': len(table.rows),
                    'row_pages': {str(row.number): row.page for row in rows if row.page},
                    'locator': {'section_ordinal': table.section_ordinal, 'table_id': table.table_id,
                                'chunk_in_table': position},
                    'boxes': [{'ref': part.ref, 'page_no': part.page, 'bbox': part.bbox}
                              for part in table.parts if part.ref in refs and part.bbox and part.page],
                    'multi_page': len({part.page for part in table.parts}) > 1,
                    'oversized': token_count > limit, 'model_derived': False, 'ocr': False,
                }))
    return chunks
