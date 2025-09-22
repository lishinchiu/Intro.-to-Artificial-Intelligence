#HW0-3
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import tempfile

image_path = 'image.png'
original_image = Image.open(image_path)
augmentations = [
    ("Original", original_image),
    ("Translation", original_image.transpose(Image.FLIP_LEFT_RIGHT)),
    ("Rotation", original_image.rotate(180)),
    ("Flipping", ImageOps.mirror(original_image)),
    ("Scaling", original_image.resize((int(original_image.width*0.1), int(original_image.height*0.1)))),
    ("Cropping", original_image.crop((608, 505, 721, 616)))
]

from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)
count=0
for title, img in augmentations:

    temp_file = tempfile.NamedTemporaryFile(suffix='.png')
    img.save(temp_file.name)

    pdf.cell(200, 20, txt=title, ln=True, align="C")
    pdf.image(temp_file.name, x=20, y=pdf.get_y() + 10, w=170)
    pdf.ln(100)
    
    count+=1
    if count==2:
      count=0
      pdf.ln(90)

pdf.output("hw0_110652036_3.pdf")
