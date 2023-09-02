import torch
import json
from transformers import CLIPModel, CLIPProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "./clip-vit-base-patch32"
model = CLIPModel.from_pretrained(repo).to(DEVICE)
processor = CLIPProcessor.from_pretrained(repo)

def get_vector(keyword):
    keyword = processor(text=keyword, return_tensors="pt")

    with torch.inference_mode():
        vectors = model.get_text_features(**keyword)
    return vectors.squeeze(0)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        keyword = body['keyword']

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json; charset=utf-8"
            },
            "body": json.dumps({
                "keyword": keyword,
                "vector": get_vector(keyword).tolist()
            }, ensure_ascii=False)
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 400,
        }