# methods for converting pdf to base64 images

# from pdf2image_base64_conversion import pdf_to_base64_pdf2image   # method 1
from pymupdf_base64_conversion import pdf_to_base64_pymupdf     # method 2

import time
import base64
import requests
import os
from dotenv import load_dotenv
from mistralai import Mistral


load_dotenv()
#optional backup encoding for single image


# def encode_image(image_path):
#     """Encode the image to base64."""
#     try:
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
#     except FileNotFoundError:
#         print(f"Error: The file {image_path} was not found.")
#         return None
#     except Exception as e:  # Added general exception handling
#         print(f"Error: {e}")
#         return None


pdf_path = "sample1.pdf"

base64_images = pdf_to_base64_pymupdf(pdf_path)  # using method 2

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

model = "pixtral-12b-2409"

client = Mistral (api_key=MISTRAL_API_KEY)

output_text_from_pdf = ""

for image in base64_images:

    messages = [

        {
            "role":"user",
            "content": [

                {"type": "text", "text": "Extract the Handwritten text in the given image"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image}"}
            ]

        }
    ]

    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    
    output_text_from_pdf+= '\n'
    output_text_from_pdf+= chat_response.choices[0].message.content

    time.sleep(5)

print(output_text_from_pdf)