"""Produce heading-aware text chunks with source provenance.

Sections follow the document's heading hierarchy; chunks pack whole blocks up to a token
target and only split a block (by sentence, then by word) when it alone exceeds the limit.
Pages come only from parser provenance: DOCX output has none, so its chunks carry a
section/block locator instead of invented page numbers. Tables are collected per section
here (merging page continuations) and chunked by table_chunker.

Legal units: Docling often leaves clause numbers inside ordinary text blocks, so each block is also
checked for a legal marker (Article/Chapter, Clause/Section/Rule, 5. / 5.1 / 5.1.2, (a) / (iv),
WHEREAS recitals, NOW THEREFORE, Schedule/Annexure, IN WITNESS WHEREOF, Hindi equivalents).
A chunk never spans two top-level units (an article, a clause or top number, a recital, a schedule);
sub-clauses stay with their clause while they fit. Every chunk records the clause path it sits in
(e.g. Clause 5 > 5.2 > (b)) and puts it in the embedding text, so a piece split from a long clause
still says which clause it belongs to.
"""
from dataclasses import dataclass, field
import re
from typing import Callable

from . import IndexChunk
from .table_chunker import LogicalTable, add_part, continues, read_part

CHUNKER_VERSION = 'legal-v1'
HEADINGS = {'title', 'section_header'}
SKIPPED = {'page_header', 'page_footer', 'picture', 'document_index'}
# Blocks that may sit between two parts of one table split across a page break.
PAGE_FURNITURE = {'page_header', 'page_footer'}
NOTES = {'visual_description': '[Model-derived image interpretation; verify against the source image]',
         'picture_ocr': '[Text read from an image by OCR]'}
SENTENCE = re.compile(r'(?<=[.!?;:])\s+')
# A period after these does not end a sentence (Rs. 5,000; No. 12; S. 138; Smt. Devi; u/s. 34; i.e.).
ABBREVIATION = re.compile(r'(?:\b(?:rs|no|nos|sr|smt|shri|sh|kum|mr|mrs|ms|dr|ltd|pvt|co|inc|sec|secs|s|art|arts|cl|'
                          r'para|paras|vs|v|viz|i\.e|e\.g|etc|hon|ld|govt|dept|st|ors|anr|w\.e\.f|r/w|u/s)|\b[A-Z])\.$', re.I)

NUM = r'[0-9\u0966-\u096f]'  # ASCII and Devanagari digits
ROMAN = r'[ivxlc]+'
# (kind, pattern, depth or None for depth from the number). Lower depth = higher level; depth <= TOP starts a chunk.
LEGAL_MARKERS = [
    ('execution', re.compile(r'^(?:in witness where ?of|signed,? sealed and delivered|जिसके साक्ष्य में)', re.I), 1),
    ('operative', re.compile(r'^now,? (?:therefore|this (?:deed|agreement|indenture) witnesse?th|it is hereby agreed)', re.I), 1),
    ('schedule', re.compile(rf'^(?:the )?(?:schedule|annexure|annex|exhibit|appendix|अनुसूची|परिशिष्ट)'
                            rf'(?:[\s\-–:]+(?:[A-Z]|{NUM}{{1,3}}|{ROMAN}))?\b', re.I), 1),
    ('article', re.compile(rf'^(?:article|chapter|part|अध्याय|भाग|अनुच्छेद)\s+(?:{NUM}{{1,3}}|{ROMAN})\b', re.I), 1),
    ('recital', re.compile(r'^(?:and )?whereas\b', re.I), 2),
    ('clause', re.compile(rf'^(?:clause|section|rule|regulation|order|धारा|खंड|नियम)\s+{NUM}{{1,3}}(?:\.{NUM}{{1,3}})*\b', re.I), None),
    ('numbered', re.compile(rf'^{NUM}{{1,3}}(?:\.{NUM}{{1,3}})+\.?(?=\s)|^{NUM}{{1,3}}[.)](?=\s)'), None),
    ('subclause', re.compile(rf'^\((?:[a-z]{{1,2}}|{ROMAN}|{NUM}{{1,2}})\)(?=\s)', re.I), 5),
]
TOP = 2
# Docling merges consecutive lines into one paragraph, so markers also occur mid-block. Split only where it is safe:
# capitalised keywords or "4." / "(a)" after the end of a sentence, and "4.1 Capital" anywhere (not after a digit,
# comma or period, so "Rs. 4.5 Lakh" stays whole). References such as "under Section 138" are never split.
INLINE = re.compile(
    rf'(?<=[.;:])\s+(?=(?:AND )?WHEREAS\b|NOW,? (?:THEREFORE|THIS)\b|IN WITNESS WHERE ?OF\b|'
    rf'(?:SCHEDULE|ANNEXURE|EXHIBIT|APPENDIX)\b|{NUM}{{1,3}}[.)]\s+["“(A-Z\u0900-\u097f]|'
    rf'\((?:[a-z]{{1,2}}|{ROMAN}|{NUM}{{1,2}})\)\s)'
    rf'|(?<![\d,.])\s+(?={NUM}{{1,3}}(?:\.{NUM}{{1,3}})+\.?\s+["“(A-Z\u0900-\u097f])')
DEFINED_TERM = re.compile(r'["“\']([^"”\']{2,60})["”\']\s+(?:shall\s+)?(?:means?|includes?|refers? to|shall have the meaning)', re.I)


@dataclass
class Unit:
    text: str
    ref: str
    label: str
    provenance: list[dict]
    position: int = 0
    legal_path: list[str] = field(default_factory=list)
    legal_kind: str | None = None  # set on the unit that starts a legal unit, not on its continuations
    legal_depth: int | None = None


def legal_marker(text):
    """(kind, label, depth) when a block starts a legal unit, else None."""
    head = text.lstrip()[:120]
    for kind, pattern, depth in LEGAL_MARKERS:
        match = pattern.match(head)
        if not match:
            continue
        label = match.group(0).strip().rstrip('.)').strip() if kind not in {'recital', 'operative', 'execution'} else kind.title()
        if kind == 'subclause':
            label = match.group(0)
        if depth is None:  # Clause 5 / 5. -> 2, 5.1 -> 3, 5.1.2 -> 4
            number = re.search(rf'{NUM}+(?:\.{NUM}+)*', match.group(0)).group(0)
            depth = min(4, 1 + number.count('.') + 1)
        return kind, label, depth
    return None


def is_roman_item(label, top):
    """Whether '(i)'-style label is a roman sub-item under `top` rather than the next letter after it."""
    inner, previous = label[1:-1].lower(), top[1][1:-1].lower()
    if not re.fullmatch(ROMAN, inner):
        return False
    if top[0] == 6:
        return True  # already inside a roman list: (ii) after (i)
    if re.fullmatch(ROMAN, previous) and roman_value(inner) == roman_value(previous) + 1 and previous != 'h':
        return False  # (ii) after (i) in a list that is roman at the top level
    # (i) after (h) and (c) after (b) are the next letter; (i) after (a) opens a roman sub-list.
    return len(previous) == 1 and previous.isalpha() and chr(ord(previous) + 1) != inner


def roman_value(text):
    values = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100}
    total = 0
    for current, following in zip(text, text[1:] + ' '):
        total += -values[current] if values.get(following, 0) > values[current] else values[current]
    return total


def assign_legal_paths(units):
    """Give each unit the clause path it sits in; a section starts with an empty path."""
    stack = []
    for unit in units:
        if unit.label in NOTES:  # image descriptions and OCR never open or close a clause
            unit.legal_path = [label for _, label in stack]
            continue
        marker = legal_marker(unit.text)
        if marker:
            kind, label, depth = marker
            if kind == 'subclause' and stack and stack[-1][0] >= 5 and is_roman_item(label, stack[-1]):
                depth = 6  # (i), (ii) nested inside (a)
            stack =[item for item in stack if item[0] < depth] + [(depth, label)]
            unit.legal_kind, unit.legal_depth = kind, depth
        unit.legal_path = [label for _, label in stack]


def split_inline_units(text):
    """One merged paragraph -> one piece per legal unit it contains."""
    pieces = []
    for piece in INLINE.split(text):
        if pieces and ABBREVIATION.search(pieces[-1]):
            pieces[-1] += ' ' + piece  # "flat No." + "4. ..." was not a clause boundary
        else:
            pieces.append(piece)
    return pieces


def split_sentences(text):
    merged = []
    for piece in SENTENCE.split(text):
        if merged and ABBREVIATION.search(merged[-1]):
            merged[-1] += ' ' + piece
        else:
            merged.append(piece)
    return merged


@dataclass
class Section:
    ordinal: int
    heading_path: list[str]
    units: list[Unit] = field(default_factory=list)
    tables: list[LogicalTable] = field(default_factory=list)

    @property
    def content(self):
        parts = [(unit.position, unit.text) for unit in self.units] + [(table.position, table.markdown()) for table in self.tables]
        return '\n\n'.join(text for _, text in sorted(parts, key=lambda item: item[0]))

    @property
    def pages(self):
        return sorted(set(pages_of(self.units)) | {page for table in self.tables for page in table.pages})


def pages_of(units):
    return sorted({item['page_no'] for unit in units for item in unit.provenance
                   if isinstance(item.get('page_no'), int) and item['page_no'] >= 1})


def build_sections(blocks, captions: dict[str, str] | None = None):
    """Group blocks under their heading path.

    Returns (sections, skipped-label counts, logical tables). A table part directly following
    another table (only page furniture in between) is merged when table_chunker.continues agrees.
    """
    captions = captions or {}
    sections, stack, skipped, tables = [], [], {}, []
    current, previous_table = Section(0, []), None
    for position, block in enumerate(blocks):
        text = (block.text or '').strip()
        if block.label == 'table':
            part = read_part(block)
            if part is None:
                skipped['table'] = skipped.get('table', 0) + 1
                continue
            if previous_table is not None and continues(previous_table, part):
                add_part(previous_table, part)
                continue
            table = LogicalTable(part.ref, 0, [item[1] for item in stack], position, part.columns, part.header,
                                 caption=captions.get(block.ref, ''))
            add_part(table, part)
            current.tables.append(table)
            tables.append(table)
            previous_table = table
            continue
        if block.label in PAGE_FURNITURE:
            skipped[block.label] = skipped.get(block.label, 0) + 1
            continue
        previous_table = None
        if block.label in HEADINGS:
            if not text:
                continue
            level = 0 if block.label == 'title' else max(1, block.level or 1)
            stack = [item for item in stack if item[0] < level] + [(level, text)]
            if current.units or current.tables:
                sections.append(current)
            current = Section(len(sections), [item[1] for item in stack])
            continue
        if block.label in SKIPPED or not text:
            if block.label in SKIPPED:
                skipped[block.label] = skipped.get(block.label, 0) + 1
            continue
        if block.label in NOTES:
            current.units.append(Unit(NOTES[block.label] + '\n' + text, block.ref, block.label, list(block.provenance), position))
            continue
        current.units.extend(Unit(piece, block.ref, block.label, list(block.provenance), position)
                             for piece in split_inline_units(text))
    if current.units or current.tables:
        sections.append(current)
    for ordinal, section in enumerate(sections):
        section.ordinal = ordinal
        for table in section.tables:
            table.section_ordinal, table.heading_path = ordinal, section.heading_path
    return sections, skipped, tables


def split_unit(unit: Unit, limit: int, count: Callable[[str], int]) -> list[Unit]:
    if count(unit.text) <= limit:
        return [unit]
    pieces, buffer = [], ''
    for sentence in split_sentences(unit.text):
        candidate = f'{buffer} {sentence}'.strip()
        if buffer and count(candidate) > limit:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)
    result = []
    for piece in pieces:
        while count(piece) > limit:
            words = piece.split()
            # Shrink by words until the head fits; guarantees progress on very long sentences.
            cut = max(1, len(words) * limit // max(count(piece), 1))
            while cut > 1 and count(' '.join(words[:cut])) > limit:
                cut = max(1, cut * 9 // 10)
            result.append(' '.join(words[:cut]))
            piece = ' '.join(words[cut:])
        if piece:
            result.append(piece)
    return [Unit(text, unit.ref, unit.label, unit.provenance, unit.position, unit.legal_path,
                 unit.legal_kind if n == 0 else None, unit.legal_depth if n == 0 else None)
            for n, text in enumerate(result)]


def chunk_sections(sections: list[Section], count: Callable[[str], int], target: int, limit: int) -> list[IndexChunk]:
    chunks = []
    for section in sections:
        heading = ' > '.join(section.heading_path)
        assign_legal_paths(section.units)
        units = [piece for unit in section.units for piece in split_unit(unit, limit, count)]
        groups, current, used = [], [], 0
        table_positions = [table.position for table in section.tables]
        for unit in units:
            size = count(unit.text)
            # Never pack text from both sides of a table into one chunk: keeps reading order and citations local.
            crosses_table = current and any(current[-1].position < position < unit.position for position in table_positions)
            # A new article, clause, recital or schedule always starts a new chunk.
            new_unit = unit.legal_depth is not None and unit.legal_depth <= TOP
            if current and (used + size > target or crosses_table or new_unit):
                groups.append(current)
                current, used = [unit], size
            else:
                current.append(unit)
                used += size
        if current:
            groups.append(current)
        for position, group in enumerate(groups):
            content = '\n\n'.join(unit.text for unit in group)
            pages = pages_of(group)
            legal_path = group[0].legal_path
            context = ' > '.join(([heading] if heading else []) + legal_path)
            defined = list(dict.fromkeys(match.group(1).strip() for unit in group for match in DEFINED_TERM.finditer(unit.text)))
            boxes = [{'ref': unit.ref, 'page_no': item['page_no'], 'bbox': item['bbox']}
                     for unit in group for item in unit.provenance if 'bbox' in item and 'page_no' in item]
            chunks.append(IndexChunk(
                ordinal=len(chunks), section_ordinal=section.ordinal, heading_path=section.heading_path,
                content=content, embedding_text=f'{context}\n\n{content}' if context else content,
                token_count=count(content), page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None, position=group[0].position,
                source_metadata={
                    'chunker_version': CHUNKER_VERSION,
                    'block_refs': list(dict.fromkeys(unit.ref for unit in group)),
                    'labels': sorted({unit.label for unit in group}),
                    'locator': {'section_ordinal': section.ordinal, 'chunk_in_section': position},
                    'boxes': boxes,
                    'model_derived': any(unit.label == 'visual_description' for unit in group),
                    'ocr': any(unit.label == 'picture_ocr' for unit in group),
                    'legal_path': legal_path,
                    'legal_units': [{'kind': unit.legal_kind, 'path': unit.legal_path} for unit in group if unit.legal_kind],
                    'defined_terms': defined,
                }))
    return chunks
