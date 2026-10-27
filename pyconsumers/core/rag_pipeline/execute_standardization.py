"""Parse and persist artifacts, with a fenced lease and atomic completion/outbox write."""
import asyncio
from contextlib import suppress
from hashlib import sha256
from io import BytesIO
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image

from ...helpers.parsing_repository import LeaseLost, ParsingRepository
from .connectors.unstructured.utils.parser import DoclingParser, ParseFailure
from .connectors.unstructured.utils.vision import VisionEnricher
from .schemas.parsed import ParsedArtifact, ParsedAsset, ParsedBlock, ParserOutput
from .userallocation.UserAllocation import StageError, StageResult


class Standardization:
    def __init__(self, db, storage, cache, settings, parser=None, vision=None):
        self.repo = ParsingRepository(db, settings)
        self.storage, self.cache, self.settings = storage, cache, settings
        self.parser = parser or DoclingParser(settings)
        self.vision = vision or VisionEnricher(settings, cache)

    async def notify(self, event):
        with suppress(Exception):
            async with asyncio.timeout(1):
                await self.cache.client.publish(self.cache.key(event.project_id, 'document-progress', event.doc_id), 'changed')

    async def execute(self, event):
        owner = str(uuid4())
        state, row = await self.repo.claim(event, owner)
        if state != 'claimed':
            return state
        await self.notify(event)
        work = asyncio.create_task(self._parse(event, owner, row))
        heartbeat = asyncio.create_task(self._heartbeat(event, owner))
        try:
            done, _ = await asyncio.wait({work, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                # Failed heartbeats cannot leave an unfenced parser publishing results.
                heartbeat.result()
                raise LeaseLost()
            manifest_key, summary = work.result()
            await self.repo.complete(event, owner, manifest_key, summary)
            await self.notify(event)
            return 'parsed'
        except LeaseLost:
            return 'busy'
        except Exception as error:
            state = await self.repo.failure(event, owner, type(error).__name__, terminal=isinstance(error, ValueError))
            await self.notify(event)
            return 'retry' if state == 'pending' else 'failed'
        finally:
            for task in (work, heartbeat):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

    async def _heartbeat(self, event, owner):
        while True:
            await asyncio.sleep(self.settings.parse_lease_seconds / 3)
            try:
                await self.repo.heartbeat(event, owner)
            except Exception:
                raise LeaseLost() from None

    async def _parse(self, event, owner, row):
        if row['byte_count'] > self.settings.parse_max_file_bytes:
            raise ValueError('File size exceeds parser limit.')
        raw = await self.storage.get_bytes(row['object_key'])
        if len(raw) > self.settings.parse_max_file_bytes or sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Raw file integrity check failed.')
        extension = '.pdf' if row['media_type'] == 'application/pdf' else '.docx'
        if row['media_type'] not in {'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}:
            raise ValueError('Unsupported document format.')
        with TemporaryDirectory(prefix='lexaior-parse-') as directory:
            directory = Path(directory)
            source, output = directory / ('source' + extension), directory / 'parsed'
            source.write_bytes(raw)
            result = await self.parser.parse(source, output)
            result = await self.vision.enrich(result, output, event.project_id)
            prefix = f'projects/{event.project_id}/documents/{event.doc_id}/parsed/{owner}'
            manifest_key, markdown_key = f'{prefix}/document.json', f'{prefix}/document.md'
            manifest = ParsedArtifact(**result.model_dump(), doc_id=event.doc_id, project_id=event.project_id,
                                      source_sha256=row['sha256'], source_filename=row['filename'],
                                      markdown_key=markdown_key, manifest_key=manifest_key)
            for asset in manifest.assets:
                local_path = output / asset.path
                await self.storage.put_bytes(f'{prefix}/{asset.path}', local_path.read_bytes(), asset.media_type)
                asset.path = f'{prefix}/{asset.path}'
            await self.storage.put_bytes(markdown_key, (output / 'document.md').read_bytes(), 'text/markdown; charset=utf-8')
            await self.storage.put_bytes(manifest_key, manifest.model_dump_json().encode(), 'application/json')
            summary = {'parser': result.parser, 'parser_version': result.parser_version,
                       'page_count': len(result.pages) if extension == '.pdf' else None,
                       'block_count': len(result.blocks),
                       'table_count': sum(block.label == 'table' for block in result.blocks),
                       'image_count': len(result.assets), 'warnings': result.warnings}
            summary['visual_description_count'] = sum(asset.visual_description is not None for asset in result.assets)
            summary['visual_interpreted_count'] = sum(asset.visual_description is not None and
                asset.visual_description.observation.status == 'interpreted' for asset in result.assets)
            summary['visual_insufficient_count'] = sum(asset.visual_description is not None and
                asset.visual_description.observation.status == 'insufficient_detail' for asset in result.assets)
            summary['vision_model'] = self.settings.vision_model if summary['visual_description_count'] else None
            return manifest_key, summary


# ---- Folder pipeline (design doc): standardization stage handler ----------------------------

IMAGE_TYPES = {'jpg': 'image/jpeg', 'png': 'image/png'}


def decode_text(raw):
    """Decode BOM-marked Unicode first, then UTF-8 and Windows text."""
    if raw.startswith(b'\xef\xbb\xbf'):
        encodings = ['utf-8-sig']
    elif raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        encodings = ['utf-16']
    else:
        encodings = ['utf-8', 'cp1252']
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeError:
            continue
        if '\x00' in text:
            raise StageError('Text contains NUL characters.', terminal=True)
        return text, encoding
    raise StageError('Text encoding could not be decoded.', terminal=True)


async def convert_to_pdf(source, directory, settings):
    """Run an isolated LibreOffice conversion, killing interrupted work."""
    process = await asyncio.create_subprocess_exec(
        settings.libreoffice_path, '--headless', '--norestore',
        f'-env:UserInstallation={(directory / "lo-profile").as_uri()}',
        '--convert-to', 'pdf', '--outdir', str(directory), str(source),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    try:
        await asyncio.wait_for(process.communicate(), settings.conversion_timeout_seconds)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    pdf = directory / f'{source.stem}.pdf'
    if not pdf.is_file():
        raise StageError('LibreOffice produced no PDF; the document may be damaged.', terminal=True)
    return pdf


def presentation_quality(source, result):
    """Detect missing text and missing chart/SmartArt images."""
    from pptx import Presentation

    slides = Presentation(source).slides
    visual_pages, visual_slides, smartart, has_text = [], [], False, False
    # Native provenance uses original slide numbers; PDF pages skip hidden slides.
    page = 0
    for number, slide in enumerate(slides, 1):
        if slide.element.get('show') in {'0', 'false'}:
            continue
        page += 1
        tags = [element.tag.rsplit('}', 1)[-1] for element in slide.element.iter()]
        if 'chart' in tags or 'relIds' in tags:
            visual_pages.append(page)
            visual_slides.append(number)
        smartart |= 'relIds' in tags
        has_text |= any(element.text and element.text.strip() for element in slide.element.iter()
                        if element.tag.endswith('}t'))
    if result is None:
        return 'Native PPTX conversion failed.', visual_pages
    if has_text and not any(block.text.strip() for block in result.blocks):
        return 'Native PPTX output omitted slide text.', visual_pages
    if smartart:
        return 'SmartArt requires rendered slide images.', visual_pages
    available = {p.get('page_no') for asset in result.assets for p in asset.provenance}
    if any(slide not in available for slide in visual_slides) or (visual_pages and
            any(note.startswith('Image unavailable') for note in result.warnings)):
        return 'Native PPTX output omitted chart images.', visual_pages
    return None, visual_pages


class FolderStandardization:
    """STNDZ stage: raw file -> markdown + manifest + images in storage; decides markdown/image next.

    PDF/DOCX/PPTX/XLSX use Docling; DOC is converted to PDF first. TXT is decoded directly.
    Image RAG handles vision separately. JPG/PNG become the
    file's only image and markdown is INAPPLICABLE. The manifest (document.json) is the parser output
    the markdown and image stages read; its key is in processing_summary.standardization.manifest_key.
    """

    def __init__(self, storage, settings, parser=None):
        self.storage, self.settings = storage, settings
        self.parser = parser or DoclingParser(settings)

    async def parse_presentation(self, source, output, directory):
        try:
            result = await self.parser.parse(source, output)
        except ParseFailure:
            result = None
        reason, visual_pages = await asyncio.to_thread(presentation_quality, source, result)
        if reason is None:
            return result, output
        pdf = await convert_to_pdf(source, directory, self.settings)
        output = directory / 'parsed-pdf'
        result = await self.parser.parse(pdf, output, render_pages=visual_pages)
        result.warnings.append(f'PPTX used LibreOffice PDF fallback: {reason} Bullet structure may be less precise.')
        return result, output

    async def __call__(self, claim):
        row = claim.row
        file_type = row['file_type']
        prefix = row['s3_path'].rsplit('/', 1)[0] + '/parsed'
        raw = await self.storage.get_bytes(row['s3_path'])
        if sha256(raw).hexdigest() != row['sha256']:
            raise StageError('Stored file does not match its upload checksum.', terminal=True)
        markdown_key = None
        if file_type in IMAGE_TYPES:
            try:
                with Image.open(BytesIO(raw)) as image:
                    image.verify()
            except Exception:
                raise StageError('The image file is damaged or not a real image.', terminal=True) from None
            result = ParserOutput(parser='image', parser_version='1', blocks=[], docling_document={}, assets=[
                ParsedAsset(ref='#/pictures/0', path=row['s3_path'], media_type=IMAGE_TYPES[file_type])])
        elif file_type == 'txt':
            text, _ = decode_text(raw)
            blocks = [ParsedBlock(ref=f'#/texts/{n}', label='text', text=paragraph.strip())
                      for n, paragraph in enumerate(re.split(r'\n[ \t\r]*\n', text)) if paragraph.strip()]
            result = ParserOutput(parser='text', parser_version='1', blocks=blocks, docling_document={})
            markdown_key = f'{prefix}/document.md'
            await self.storage.put_bytes(markdown_key, text.encode('utf-8'), 'text/markdown; charset=utf-8')
        elif file_type in {'pdf', 'docx', 'doc', 'pptx', 'xlsx'}:
            with TemporaryDirectory(prefix='lexaior-stndz-') as directory:
                directory = Path(directory)
                source, output = directory / f'source.{file_type}', directory / 'parsed'
                source.write_bytes(raw)
                if file_type == 'doc':
                    source = await convert_to_pdf(source, directory, self.settings)
                if file_type == 'pptx':
                    result, output = await self.parse_presentation(source, output, directory)
                else:
                    result = await self.parser.parse(source, output)
                for asset in result.assets:
                    key = f'{prefix}/{asset.path}'
                    await self.storage.put_bytes(key, (output / asset.path).read_bytes(), asset.media_type)
                    asset.path = key
                markdown_key = f'{prefix}/document.md'
                await self.storage.put_bytes(markdown_key, (output / 'document.md').read_bytes(), 'text/markdown; charset=utf-8')
        else:
            raise StageError(f'Unsupported file type: {file_type}.', terminal=True)
        manifest_key = f'{prefix}/document.json'
        await self.storage.put_bytes(manifest_key, result.model_dump_json().encode(), 'application/json')
        has_text = any(block.text.strip() for block in result.blocks)
        summary = {'parser': result.parser, 'parser_version': result.parser_version, 'manifest_key': manifest_key,
                   'block_count': len(result.blocks), 'table_count': sum(b.label == 'table' for b in result.blocks),
                   'image_count': len(result.assets), 'warnings': result.warnings}
        return StageResult(
            updates={'s3_path_tomarkdown': markdown_key, 's3_path_images': f'{prefix}/assets' if result.assets else None,
                     'page_count': len(result.pages) if file_type in {'pdf', 'doc', 'pptx'} else None},
            summary=summary,
            next={'markdown': 'WAITING' if has_text else 'INAPPLICABLE',
                  'image': 'WAITING' if result.assets else 'INAPPLICABLE'})
