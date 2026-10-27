"""Generate small fictional document fixtures; no customer documents or external assets."""
from io import BytesIO
from pathlib import Path
import os
import subprocess
from tempfile import TemporaryDirectory

from docx import Document
from docx.shared import Inches
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def generate(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    normal = folder / 'normal.pdf'
    pdf = canvas.Canvas(str(normal))
    pdf.setFont('Helvetica-Bold', 20)
    pdf.drawString(50, 780, 'Lexaior sample guideline')
    pdf.setFont('Helvetica', 12)
    pdf.drawString(50, 745, 'The recommended review interval is fourteen days.')
    pdf.showPage()
    pdf.setFont('Helvetica', 12)
    pdf.drawString(50, 780, 'The second page describes the follow-up appointment.')
    pdf.save()

    # Render a scan with image pixels only; there is no PDF text layer.
    scan = Image.new('RGB', (1600, 1000), 'white')
    draw = ImageDraw.Draw(scan)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    draw.text((80, 120), 'SCANNED MEDICAL GUIDELINE', fill='black', font=font)
    draw.text((80, 230), 'Review the patient after fourteen days.', fill='black', font=font)
    draw.text((80, 340), 'Document number 12345.', fill='black', font=font)
    scan.save(folder / 'scan.png')
    pdf = canvas.Canvas(str(folder / 'scanned.pdf'), pagesize=(800, 500))
    pdf.drawImage(ImageReader(scan), 0, 0, width=800, height=500)
    pdf.save()

    # A clear tabular PDF with a separate chart-like graphic.
    chart = Image.new('RGB', (600, 350), 'white')
    drawing = ImageDraw.Draw(chart)
    drawing.line((40, 20, 40, 310, 580, 310), fill='black', width=5)
    for x, height, color in [(100, 100, 'blue'), (250, 200, 'green'), (400, 260, 'orange')]:
        drawing.rectangle((x, 310-height, x+80, 310), fill=color)
    chart.save(folder / 'chart.png')
    pdf = canvas.Canvas(str(folder / 'table_image.pdf'))
    pdf.setFont('Helvetica-Bold', 18)
    pdf.drawString(50, 790, 'Treatment schedule')
    for index, values in enumerate([('Treatment', 'Interval', 'Duration'), ('Exercise', 'Daily', '14 days'),
                                     ('Review', 'Weekly', '28 days')]):
        y=730-index*35
        for column, value in enumerate(values):
            pdf.setFont('Helvetica-Bold' if index==0 else 'Helvetica', 12)
            pdf.drawString(55+column*160, y, value)
    for y in [750, 715, 680, 645]:
        pdf.line(50, y, 530, y)
    for x in [50, 210, 370, 530]:
        pdf.line(x, 645, x, 750)
    pdf.drawImage(ImageReader(chart), 100, 280, width=400, height=230)
    pdf.setFont('Helvetica', 12)
    pdf.drawString(100, 250, 'Figure 1. Fictional chart for image extraction testing.')
    pdf.save()

    document = Document()
    document.add_heading('Lexaior sample document', 0)
    document.add_heading('Follow-up guidance', 1)
    document.add_paragraph('The recommended review interval is fourteen days.')
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = 'Treatment', 'Interval'
    cells = table.add_row().cells
    cells[0].text, cells[1].text = 'Exercise', 'Daily'
    document.add_picture(str(folder / 'scan.png'), width=Inches(5))
    document.save(folder / 'sample.docx')
    (folder / 'corrupt.pdf').write_bytes(b'%PDF-1.7\nnot a valid document')
    generate_formats(folder)
    generate_long_table(folder)
    generate_lease_deed(folder)
    generate_court_order(folder)


def generate_formats(folder: Path):
    """Text, tabular and Office examples containing only fictional data."""
    from openpyxl import Workbook
    from pptx import Presentation

    text = 'Lexaior fictional guideline\n\nReview the matter after fourteen days.\n\nन्यायालय में अगली सुनवाई चौदह दिनों के बाद होगी।\n'
    (folder / 'sample.txt').write_text(text, encoding='utf-8')
    (folder / 'sample_bom.txt').write_text(text, encoding='utf-8-sig')
    (folder / 'sample.csv').write_text('Party,Amount,Code\nAsha Verma,15000,001\nRavi Kumar,25000,002\n', encoding='utf-8')
    presentation = Presentation()
    for title, body in [('Fictional review guideline', 'Review the matter after fourteen days.'),
                        ('Fictional payment schedule', 'Asha Verma: Rs. 15000. Ravi Kumar: Rs. 25000.')]:
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    presentation.save(folder / 'sample.pptx')
    generate_visual_presentation(folder)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Fictional schedule'
    for row in [('Party', 'Amount', 'Code'), ('Asha Verma', 15000, '001'), ('Ravi Kumar', 25000, '002')]:
        sheet.append(row)
    workbook.save(folder / 'sample.xlsx')
    soffice = os.environ.get('LIBREOFFICE_PATH', '/Applications/LibreOffice.app/Contents/MacOS/soffice')
    with TemporaryDirectory(prefix='lexaior-fixture-') as directory:
        subprocess.run([soffice, '--headless', '--norestore',
                        f'-env:UserInstallation={(Path(directory) / "lo-profile").as_uri()}',
                        '--convert-to', 'doc:MS Word 97', '--outdir', str(folder), str(folder / 'sample.docx')],
                       check=True, timeout=180)
    if not (folder / 'sample.doc').is_file():
        raise RuntimeError('LibreOffice produced no DOC fixture.')


def generate_visual_presentation(folder: Path):
    """A native chart and a process diagram, with no embedded raster image."""
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches as PptInches

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = 'Fictional rent review'
    chart = CategoryChartData()
    chart.categories = ['Asha', 'Ravi']
    chart.add_series('Monthly rent', [15000, 25000])
    slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, PptInches(1), PptInches(1.4),
                           PptInches(8), PptInches(4), chart)
    for x, label, shape in [(1, 'File received', MSO_SHAPE.ROUNDED_RECTANGLE),
                            (4, '', MSO_SHAPE.RIGHT_ARROW), (6, 'Review in fourteen days', MSO_SHAPE.ROUNDED_RECTANGLE)]:
        item = slide.shapes.add_shape(shape, PptInches(x), PptInches(5.7), PptInches(2.5), PptInches(1))
        item.text = label
    presentation.save(folder / 'sample_visual.pptx')


def generate_long_table(folder: Path):
    """A fictional 55-row expense statement that spans two pages and repeats its header."""
    header = ('Date', 'Provider', 'Description', 'Amount (INR)')
    rows = [(f'{(n % 28) + 1:02d}/0{(n // 28) + 1}/2026', f'Provider {n % 7 + 1}', f'Treatment item {n}', f'{(n + 1) * 150}')
            for n in range(55)]
    pdf = canvas.Canvas(str(folder / 'long_table.pdf'))
    widths, left, height = (90, 120, 170, 100), 50, 18

    def draw_table(top, items):
        lines = [header] + list(items)
        for index, values in enumerate(lines):
            y = top - (index + 1) * height
            x = left
            for width, value in zip(widths, values):
                pdf.setFont('Helvetica-Bold' if index == 0 else 'Helvetica', 9)
                pdf.drawString(x + 4, y + 5, value)
                x += width
        bottom = top - len(lines) * height
        for index in range(len(lines) + 1):
            pdf.line(left, top - index * height, left + sum(widths), top - index * height)
        x = left
        for width in (0,) + widths:
            x += width
            pdf.line(x, top, x, bottom)

    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(50, 800, 'Statement of medical expenses')
    pdf.setFont('Helvetica', 10)
    pdf.drawString(50, 782, 'Fictional claim 42/2026. All amounts in rupees.')
    draw_table(770, rows[:38])
    pdf.drawString(280, 30, 'Page 1')
    pdf.showPage()
    draw_table(800, rows[38:])
    pdf.setFont('Helvetica', 10)
    pdf.drawString(50, 800 - (len(rows) - 38 + 1) * height - 30, 'Total claimed is the sum of all rows above.')
    pdf.drawString(280, 30, 'Page 2')
    pdf.save()


def generate_lease_deed(folder: Path):
    """A fictional two-page rent agreement: recitals, numbered clauses, sub-clauses, a schedule, execution."""
    lines = [
        ('h', 'RENT AGREEMENT'),
        ('p', 'Made at Patna on 1 April 2026 between Asha Verma (Lessor) and Ravi Kumar (Lessee).'),
        ('p', 'WHEREAS the Lessor is the owner of flat No. 4, Kankarbagh, Patna.'),
        ('p', 'AND WHEREAS the Lessee wishes to take the flat on rent for residential use.'),
        ('p', 'NOW THIS AGREEMENT WITNESSETH as follows:'),
        ('p', '1. Definitions'),
        ('p', '1.1 "Premises" means the flat described in Schedule A.'),
        ('p', '1.2 "Rent" means the monthly amount payable under clause 2.'),
        ('p', '2. Rent and deposit'),
        ('p', '2.1 The Lessee shall pay a monthly rent of Rs. 15,000 on or before the 5th day of each month.'),
        ('p', '2.2 The Lessee shall pay a security deposit of Rs. 45,000, refundable on termination:'),
        ('p', '(a) after deducting unpaid rent;'),
        ('p', '(b) after deducting the cost of damage, other than normal wear and tear.'),
        ('p', '3. Term and termination'),
        ('p', '3.1 This agreement is for eleven months from 1 April 2026.'),
        ('p', '3.2 Either party may terminate it by giving one month written notice.'),
        ('break', ''),
        ('p', '4. Maintenance'),
        ('p', '4.1 The Lessee shall keep the Premises clean and shall not make structural changes.'),
        ('p', '5. Governing law'),
        ('p', '5.1 This agreement is governed by the laws of India and courts at Patna have jurisdiction.'),
        ('p', 'IN WITNESS WHEREOF the parties have signed this agreement on the date above.'),
        ('p', 'SCHEDULE A'),
        ('p', 'Flat No. 4, second floor, Kankarbagh, Patna, measuring 900 square feet.'),
    ]
    pdf = canvas.Canvas(str(folder / 'lease_deed.pdf'))
    y = 790
    for kind, value in lines:
        if kind == 'break':
            pdf.showPage()
            y = 790
            continue
        pdf.setFont('Helvetica-Bold' if kind == 'h' else 'Helvetica', 16 if kind == 'h' else 11)
        pdf.drawString(50, y, value)
        y -= 34 if kind == 'h' else 24
    pdf.save()


def generate_court_order(folder: Path):
    """A fictional one-page cheque-bounce order: court, case number, judge, parties, statutes, dates, fine."""
    lines = [
        ('h', 'IN THE COURT OF THE JUDICIAL MAGISTRATE FIRST CLASS, PATNA'),
        ('p', 'Complaint Case No. 1234 of 2025'),
        ('p', 'Meera Sinha ... Complainant'),
        ('p', 'versus'),
        ('p', 'Arun Prakash ... Accused'),
        ('p', 'Present: Shri R. K. Mishra, Judicial Magistrate First Class.'),
        ('h', 'ORDER'),
        ('p', 'The complaint is filed under Section 138 of the Negotiable Instruments Act, 1881.'),
        ('p', 'The accused issued cheque No. 456789 dated 10 January 2025 for Rs. 2,50,000.'),
        ('p', 'The cheque was returned unpaid for insufficient funds on 15 January 2025.'),
        ('p', 'The accused is convicted and directed to pay compensation of Rs. 3,00,000'),
        ('p', 'to the complainant under Section 357(3) of the Code of Criminal Procedure, 1973.'),
        ('p', 'Pronounced in open court on 20 March 2026.'),
    ]
    pdf = canvas.Canvas(str(folder / 'court_order.pdf'))
    y = 790
    for kind, value in lines:
        pdf.setFont('Helvetica-Bold' if kind == 'h' else 'Helvetica', 13 if kind == 'h' else 11)
        pdf.drawString(50, y, value)
        y -= 30 if kind == 'h' else 22
    pdf.save()


if __name__ == '__main__':
    generate(Path(__file__).parent)
