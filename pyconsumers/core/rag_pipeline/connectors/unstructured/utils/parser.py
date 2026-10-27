"""Docling adapter and killable subprocess boundary for untrusted document parsing."""
import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from ....schemas.parsed import ParsedAsset, ParsedBlock, ParserOutput


class ParseFailure(Exception):
    pass


class DoclingParser:
    def __init__(self, settings):
        self.settings = settings

    async def parse(self, source: Path, output: Path, render_pages=None):
        config = {'threads': self.settings.parser_threads, 'force_ocr': self.settings.parser_force_ocr,
                  'max_pages': self.settings.parse_max_pages, 'max_file_bytes': self.settings.parse_max_file_bytes,
                  'render_pages': render_pages or []}
        cache = Path(__file__).resolve().parents[6] / '.cache'
        cache.mkdir(exist_ok=True)
        env = os.environ | {'HF_HOME': str(cache / 'huggingface'),
                            'DOCLING_CACHE_DIR': str(cache / 'docling'),
                            'HF_HUB_DISABLE_TELEMETRY': '1', 'OMP_NUM_THREADS': str(self.settings.parser_threads)}
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'pyconsumers.core.rag_pipeline.connectors.unstructured.utils.parser',
            str(source), str(output), json.dumps(config), stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE, env=env)
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self.settings.parse_timeout_seconds)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        if process.returncode:
            # Document contents and third-party tracebacks must not be stored in DB errors.
            raise ParseFailure('Docling conversion failed; check parser dependencies and document validity.')
        try:
            result = ParserOutput.model_validate_json((output / 'document.json').read_bytes())
            if not (output / 'document.md').is_file():
                raise ValueError('Missing Markdown output.')
            for asset in result.assets:
                path = Path(asset.path)
                if path.is_absolute() or '..' in path.parts or not (output / path).is_file():
                    raise ValueError('Invalid asset output.')
            return result
        except (OSError, ValueError):
            raise ParseFailure('Parser produced invalid artifacts.') from None


def convert(source, output, config):
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, OcrMode
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem, TableItem

    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(num_threads=config['threads'], device=AcceleratorDevice.CPU)
    options.do_ocr = True
    options.ocr_options = RapidOcrOptions(backend='onnxruntime', lang=['en'],
                                         mode=OcrMode.FULL_PAGE if config['force_ocr'] else OcrMode.DEFAULT)
    options.do_table_structure = True
    options.generate_picture_images = True
    options.images_scale = 1.5
    options.enable_remote_services = False
    options.do_picture_description = False
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX, InputFormat.XLSX],
                                  format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    conversion = converter.convert(source, raises_on_error=True, max_num_pages=config['max_pages'],
                                   max_file_size=config['max_file_bytes'])
    if conversion.status != ConversionStatus.SUCCESS:
        raise ParseFailure('Partial/failed conversions are not marked PARSED.')
    document = conversion.document
    output.mkdir(parents=True, exist_ok=True)
    (output / 'assets').mkdir(exist_ok=True)
    blocks, assets, notes, image_ocr = [], [], [], []
    ocr = None
    for item, level in document.iterate_items():
        provenance = [entry.model_dump(mode='json') for entry in getattr(item, 'prov', [])]
        label = str(item.label.value)
        text = getattr(item, 'text', '') or ''
        data = item.data.model_dump(mode='json') if isinstance(item, TableItem) else None
        if isinstance(item, TableItem):
            text = item.export_to_markdown(doc=document)
        marker = (getattr(item, 'marker', '') or '').strip()
        if getattr(item, 'enumerated', False) and marker and not text.startswith(marker):
            text = f'{marker} {text}'  # Docling strips clause numbers ("2.1") from list items; citations need them
        blocks.append(ParsedBlock(ref=item.self_ref, label=label, text=text,
                                  level=getattr(item, 'level', None), provenance=provenance, data=data))
        if isinstance(item, PictureItem):
            image = item.get_image(document)
            if image is None:
                notes.append(f'Image unavailable for {item.self_ref}.')
                continue
            relative = f'assets/picture-{len(assets) + 1:04d}.png'
            image.save(output / relative, format='PNG')
            ocr_text = ''
            # DOCX is parsed natively, so its embedded images need their own OCR pass.
            if source.suffix.lower() == '.docx':
                import numpy as np
                from rapidocr import RapidOCR
                if ocr is None:
                    ocr = RapidOCR()
                recognized = ocr(np.asarray(image.convert('RGB')))
                ocr_text = '\n'.join(recognized.txts or [])
                if ocr_text:
                    blocks.append(ParsedBlock(ref=f'{item.self_ref}/ocr', label='picture_ocr',
                                              text=ocr_text, provenance=provenance))
                    image_ocr.append(f'\n\n[OCR from image {item.self_ref}]\n\n{ocr_text}')
            assets.append(ParsedAsset(ref=item.self_ref, path=relative, provenance=provenance, ocr_text=ocr_text))
    if config.get('render_pages'):
        import pypdfium2 as pdfium
        rendered = set(config['render_pages'])
        assets = [asset for asset in assets if not any(p.get('page_no') in rendered for p in asset.provenance)]
        with pdfium.PdfDocument(str(source)) as pdf:
            for number in sorted(rendered):
                page = pdf[number - 1]
                try:
                    bitmap = page.render(scale=1.5)
                    try:
                        relative = f'assets/slide-{number:04d}.png'
                        bitmap.to_pil().save(output / relative, format='PNG')
                    finally:
                        bitmap.close()
                finally:
                    page.close()
                ref = f'#/rendered-slides/{number}'
                provenance = [{'page_no': number}]
                assets.append(ParsedAsset(ref=ref, path=relative, provenance=provenance))
                blocks.append(ParsedBlock(ref=ref, label='picture', provenance=provenance))
    if assets:
        notes.append('Images are extracted; charts, diagrams and photos are not semantically interpreted.')
    if source.suffix.lower() == '.docx':
        notes.append('DOCX page numbers are unavailable without document pagination; none are invented.')
    raw = document.export_to_dict()
    # Image bytes are stored as objects, not repeated as large base64 fields in JSON.
    for collection in ['pictures', 'tables']:
        for item in raw.get(collection, []):
            item.pop('image', None)
    for page in raw.get('pages', {}).values():
        page.pop('image', None)
    pages = [{'page_no': page.page_no, 'size': page.size.model_dump(mode='json')}
             for page in document.pages.values()]
    result = ParserOutput(parser_version=version('docling-slim'), pages=pages, blocks=blocks,
                          assets=assets, warnings=notes, docling_document=raw)
    (output / 'document.json').write_text(result.model_dump_json(indent=2))
    (output / 'document.md').write_text(document.export_to_markdown() + ''.join(image_ocr))


if __name__ == '__main__':
    convert(Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3]))
