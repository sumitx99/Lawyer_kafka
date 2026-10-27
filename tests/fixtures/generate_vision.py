"""Fictional labeled chart fixture for real vision-model evaluation."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', (1000, 760), 'white')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 32)
        small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 26)
    except OSError:
        font, small = ImageFont.load_default(size=32), ImageFont.load_default(size=26)
    draw.text((300, 25), 'Quarterly sales', font=font, fill='black')
    draw.text((20, 90), 'Units sold', font=small, fill='black')
    draw.line((140, 150, 140, 650, 920, 650), fill='black', width=4)
    for value in [0, 10, 20, 30]:
        y = 650 - value * 14
        draw.text((85, y-18), str(value), font=small, fill='black')
        draw.line((135, y, 920, y), fill='#cccccc', width=1)
    for x, value, label in [(250, 10, 'Q1'), (480, 20, 'Q2'), (710, 30, 'Q3')]:
        y = 650 - value * 14
        draw.rectangle((x, y, x+110, 648), fill='#2864bc')
        draw.text((x+35, y-42), str(value), font=font, fill='black')
        draw.text((x+30, 665), label, font=font, fill='black')
    draw.text((470, 715), 'Quarter', font=small, fill='black')
    image.save(folder / 'labeled_chart.png')
    pdf = canvas.Canvas(str(folder / 'labeled_chart.pdf'))
    pdf.setFont('Helvetica', 12)
    pdf.drawString(50, 790, 'Fictional example for chart interpretation testing.')
    pdf.drawImage(ImageReader(image), 40, 230, width=520, height=395)
    pdf.save()


if __name__ == '__main__':
    generate(Path(__file__).parent)
