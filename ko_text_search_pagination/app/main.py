import numpy as np
import torch
import json
import faiss
import pymongo
import os
from transformers import AutoModel, AutoProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "./clip-vit-base-patch32-ko"
model = AutoModel.from_pretrained(repo).to(DEVICE)
processor = AutoProcessor.from_pretrained(repo)

conn = pymongo.MongoClient(host=os.environ.get('MONGO_HOST'), port=int(os.environ.get('MONGO_PORT')), username=os.environ.get('MONGO_USERNAME'), password=os.environ.get('MONGO_PASSWORD'))
db = conn[os.environ.get('MONGO_DBNAME')]
koclip_index = faiss.read_index(os.environ.get('INDEX_SAVE_PATH') + '/koclip.index')
total_documents = koclip_index.ntotal

def to_json(bson: dict, score: float=None) -> dict:
    json = {}
    json['_id'] = str(bson['_id'])
    json['image'] = bson['image']
    json['cursor'] = bson['cursor']
    json['createdAt'] = str(bson['createdAt'])
    json['updatedAt'] = str(bson['updatedAt'])
    json['faiss_id'] = bson['faiss_id']
    json['tag_ins'] = bson['tag_ins']
    json['owner_store_id'] = bson['owner_store_id']
    json['user_like_ids'] = bson['user_like_ids']
    json['score'] = score

    return json

def get_vector(keyword: str) -> np.ndarray:
    keyword = processor(text=keyword, return_tensors="pt")

    with torch.inference_mode():
        vectors = model.get_text_features(**keyword)
    return vectors.numpy()

def lambda_handler(event: dict, context: dict) -> dict:
    if db.counters.find_one({'sequenceName': 'cakes'})['seq'] != total_documents:
        global koclip_index
        koclip_index = faiss.read_index(os.environ.get('INDEX_SAVE_PATH') + '/koclip.index')

    try:
        queries = event['queryStringParameters']
        keyword = queries['keyword']
        keyword_vector = get_vector(keyword)

        page = int(queries['page'])
        size = int(queries['size'])

        distances, indices = koclip_index.search(keyword_vector, total_documents)

        distances_list = distances[0][page * size: (page + 1) * size].tolist()
        faiss_ids = indices[0][page * size: (page + 1) * size].tolist()

        zip_data = list(zip(distances_list, faiss_ids))
        zip_data.sort(key=lambda x: x[1])

        cake_documents = list(zip(zip_data, list(db.cakes.find({'faiss_id' : {'$in': faiss_ids}}).sort('faiss_id', pymongo.ASCENDING))))
        cake_documents.sort(key=lambda x: -x[0][0])
        cake_documents = list(map(lambda x: to_json(x[1], x[0][0]), cake_documents))

        is_last_page = indices[0][(page + 1) * size:].size == 0

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json; charset=utf-8"
            },
            "body": json.dumps({
                "result": cake_documents,
                "totalDocuments": total_documents,
                "size": size,
                "nextPage": page + 1,
                "isLastPage": is_last_page,
            })
        }
    except Exception as e:
        print(e)
        return {
            "statusCode": 400,
        }