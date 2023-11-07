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
    json['score'] = score if score is not None else bson['score']

    return json

def get_vector(keyword: str) -> np.ndarray:
    keyword = processor(text=keyword, return_tensors="pt")

    with torch.inference_mode():
        vectors = model.get_text_features(**keyword)
    return vectors.numpy()

def get_cake_pipeline(size: int, sorted_distance: list, faiss_ids: list, count_delete: int):

    return [
		{
            '$match': {
                '$and': [
                    {'is_delete': False},
                    {'faiss_id': {'$in': faiss_ids[:(size + count_delete)]}}
                ]
            }
		},
		{
            '$project': {
                'vit': 0,
                'koclip': 0
            }
		},
		{
            '$addFields': {
                'score': {
                    '$arrayElemAt': [sorted_distance, '$faiss_id']
                }
            }
		},
		{
            '$sort': {
                'score': -1
            }
		},
		{
			'$limit': size
		}
    ]

def lambda_handler(event: dict, context: dict) -> dict:
    if db.counters.find_one({'sequenceName': 'cakes'})['seq'] + 1 != total_documents:
        global koclip_index
        global total_documents
        koclip_index = faiss.read_index(os.environ.get('INDEX_SAVE_PATH') + '/koclip.index')
        total_documents = koclip_index.ntotal

    try:
        queries = event['queryStringParameters']
        keyword = queries['keyword']
        keyword_vector = get_vector(keyword)
        size = int(queries['size'])

        count_delete = db.cakes.count_documents({'is_delete': True})

        distances, indices = koclip_index.search(keyword_vector, koclip_index.ntotal)

        distances_list = distances[0].tolist()
        faiss_ids = indices[0].tolist()

        zip_data = list(zip(distances_list, faiss_ids))
        zip_data.sort(key=lambda x: x[1])
        sorted_distances = list(map(lambda x: x[0], zip_data))

        cake_documents = list(db.cakes.aggregate(get_cake_pipeline(size, sorted_distances, faiss_ids, count_delete)))
        cake_documents = list(map(lambda x: to_json(x), cake_documents))

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json; charset=utf-8"
            },
            "body": json.dumps({
                "result": cake_documents
            })
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 400,
        }