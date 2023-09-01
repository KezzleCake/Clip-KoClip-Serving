import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from io import BytesIO

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "./clip-vit-base-patch32"
model = CLIPModel.from_pretrained(repo).to(DEVICE)
processor = CLIPProcessor.from_pretrained(repo)

def get_vector(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = processor(images=img, return_tensors="pt")

    with torch.inference_mode():
        vectors = model.get_image_features(**img)
    return vectors.squeeze(0)

def lambda_handler(event, context):
    try:
        object_url = event['url']

        return {
            "model": "clip",
            "key": event['key'],
            "vector": get_vector(object_url).tolist()
        }
    except Exception as e:
        print(e)
        raise e