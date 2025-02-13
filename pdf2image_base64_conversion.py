import pdf2image
import base64
from io import BytesIO
from PIL import Image

def pdf_to_base64_pdf2image (path:str):

    images = pdf2image.convert_from_path(path)

    base64_images = []

    for img in images:
        buffered = BytesIO()
        img.save(buffered , format = "PNG")
        img_base64 = base64.b64encode(buffered.getValue()).decode("utf-8")

        base64_images.append(img_base64)

    return base64_images


pdf_path = "sample1.pdf"

base64_images = pdf_to_base64_pdf2image(pdf_path)

print(base64_images)