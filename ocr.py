import fitz # PyMuPDF
import base64
from io import BytesIO
from PIL import Image

def pdf_to_base64( path : str):

    doc = fitz.open(path)

    base64_images = []

    for page in doc:
        pic = doc[page].get_pixmap()
        img = Image.frombytes("RGB" , [pic.width , pic.height] , pic.samples)

        buffered = BytesIO()

        img.save(buffered , format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        base64_images.append(img_base64)

    return base64_images

pdf_path = "sample1.pdf"