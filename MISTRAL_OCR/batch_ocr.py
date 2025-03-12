

# !pip install mistralai datasets

from mistralai import Mistral

api_key = "apikey"
client = Mistral(api_key=api_key)
ocr_model = "mistral-ocr-latest"

"""## Without Batch"""

import base64
from io import BytesIO
from PIL import Image

def encode_image_data(image_data):
    try:
        # Ensure image_data is bytes
        if isinstance(image_data, bytes):
            # Directly encode bytes to base64
            return base64.b64encode(image_data).decode('utf-8')
        else:
            # Convert image data to bytes if it's not already
            buffered = BytesIO()
            image_data.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

from datasets import load_dataset

n_samples = 100
dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)
subset = list(dataset.take(n_samples))

from tqdm import tqdm

ocr_dataset = []
for sample in tqdm(subset):
    image_data = sample['image']  # 'image' contains the actual image data

    # Encode the image data to base64
    base64_image = encode_image_data(image_data)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    # Process the image using Mistral OCR
    response = client.ocr.process(
        model=ocr_model,
        document={
            "type": "image_url",
            "image_url": image_url,
        }
    )

    # Store the image data and OCR content in the new dataset
    ocr_dataset.append({
        'image': base64_image,
        'ocr_content': response.pages[0].markdown # Since we are dealing with single images, there will be only one page
    })

import json

with open('ocr_dataset.json', 'w') as f:
    json.dump(ocr_dataset, f, indent=4)

"""
## With Batch

To use Batch Inference, we need to create a JSONL file containing all the image data and request information for our batch.

"""

def create_batch_file(image_urls, output_file):
    with open(output_file, 'w') as file:
        for index, url in enumerate(image_urls):
            entry = {
                "custom_id": str(index),
                "body": {
                    "document": {
                        "type": "image_url",
                        "image_url": url
                    },
                    "include_image_base64": True
                }
            }
            file.write(json.dumps(entry) + '\n')

"""The next step involves encoding the data of each image into base64 and saving the URL of each image that will be used."""

image_urls = []
for sample in tqdm(subset):
    image_data = sample['image']  # 'image' contains the actual image data

    # Encode the image data to base64 and add the url to the list
    base64_image = encode_image_data(image_data)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    image_urls.append(image_url)

"""We can now create our batch file."""

batch_file = "batch_file.jsonl"
create_batch_file(image_urls, batch_file)

batch_data = client.files.upload(
    file={
        "file_name": batch_file,
        "content": open(batch_file, "rb")},
    purpose = "batch"
)

"""The file is uploaded, but the batch inference has not started yet. To initiate it, we need to create a job."""

created_job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model=ocr_model,
    endpoint="/v1/ocr",
    metadata={"job_type": "testing"}
)

retrieved_job = client.batch.jobs.get(job_id=created_job.id)
print(f"Status: {retrieved_job.status}")
print(f"Total requests: {retrieved_job.total_requests}")
print(f"Failed requests: {retrieved_job.failed_requests}")
print(f"Successful requests: {retrieved_job.succeeded_requests}")
print(
    f"Percent done: {round((retrieved_job.succeeded_requests + retrieved_job.failed_requests) / retrieved_job.total_requests, 4) * 100}%"
)

"""Let's automate this feedback loop and download the results once they are ready!"""

import time
from IPython.display import clear_output

while retrieved_job.status in ["QUEUED", "RUNNING"]:
    retrieved_job = client.batch.jobs.get(job_id=created_job.id)

    clear_output(wait=True)  # Clear the previous output ( User Friendly )
    print(f"Status: {retrieved_job.status}")
    print(f"Total requests: {retrieved_job.total_requests}")
    print(f"Failed requests: {retrieved_job.failed_requests}")
    print(f"Successful requests: {retrieved_job.succeeded_requests}")
    print(
        f"Percent done: {round((retrieved_job.succeeded_requests + retrieved_job.failed_requests) / retrieved_job.total_requests, 4) * 100}%"
    )
    time.sleep(2)

client.files.download(file_id=retrieved_job.output_file)

"""Done! With this method, you can perform OCR tasks in bulk in a very cost-effective way."""