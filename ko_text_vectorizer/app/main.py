import torch
import json
import faiss
import pymongo
import os
import time
import numpy as np
from transformers import AutoModel, AutoProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "./clip-vit-base-patch32-ko"
model = AutoModel.from_pretrained(repo).to(DEVICE)
processor = AutoProcessor.from_pretrained(repo)

conn = pymongo.MongoClient(host=os.environ.get('MONGO_HOST'), port=int(os.environ.get('MONGO_PORT')), username=os.environ.get('MONGO_USERNAME'), password=os.environ.get('MONGO_PASSWORD'))
db = conn[os.environ.get('MONGO_DBNAME')]

koclip_index = faiss.read_index(os.environ.get('INDEX_SAVE_PATH') + '/koclip.index')

def get_vector(keyword):
    keyword = processor(text=keyword, return_tensors="pt")

    with torch.inference_mode():
        vectors = model.get_text_features(**keyword)
    return vectors.numpy()


def lambda_handler(event, context):
    try:
        queries = event['queryStringParameters']
        keyword = queries['keyword']
        keyword_vector = get_vector(keyword)
        size = int(queries['size'])

        distances, indices = koclip_index.search(keyword_vector, size)

        result_document = []


        for i, index in enumerate(indices[0]):
            cake_document = db.cakes.find_one({'faiss_id': index.item()})
            if cake_document is None:
                continue
            cake_document['_id'] = str(cake_document['_id'])
            cake_document['createdAt'] = str(cake_document['createdAt'])
            cake_document['updatedAt'] = str(cake_document['updatedAt'])
            cake_document['similiarity'] = distances[0][i].item()

            document_keys = cake_document.keys()

            if ('vit' in document_keys):
                del cake_document['vit']
            if ('koclip' in document_keys):
                del cake_document['koclip']

            result_document.append(cake_document)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json; charset=utf-8"
            },
            "body": json.dumps({
                "result": result_document
            })
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 400,
        }

# start_time = time.time()
# print(
# lambda_handler({
#     "queryStringParameters": {
#         "keyword": "딸기",
#         "size": 6
#     }
# }, None))
# print("--- 함수 실행 시간 %s seconds ---" % (time.time() - start_time))