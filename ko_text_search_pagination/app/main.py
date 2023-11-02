import time
import torch
import json
import faiss
import pymongo
import os
from operator import itemgetter
from transformers import AutoModel, AutoProcessor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

repo = "./clip-vit-base-patch32-ko"
model = AutoModel.from_pretrained(repo).to(DEVICE)
processor = AutoProcessor.from_pretrained(repo)

conn = pymongo.MongoClient(host=os.environ.get('MONGO_HOST'), port=int(os.environ.get('MONGO_PORT')), username=os.environ.get('MONGO_USERNAME'), password=os.environ.get('MONGO_PASSWORD'))
db = conn[os.environ.get('MONGO_DBNAME')]
koclip_index = faiss.read_index(os.environ.get('INDEX_SAVE_PATH') + '/koclip.index')

def to_json(bson):
    json = {}
    json['_id'] = str(bson['_id'])
    json['image'] = bson['image']
    # json['content_ins'] = None if bson['content_ins'] is None else bson['content_ins']
    json['cursor'] = bson['cursor']
    json['createdAt'] = str(bson['createdAt'])
    json['updatedAt'] = str(bson['updatedAt'])
    json['faiss_id'] = bson['faiss_id']
    # json['like_ins'] = bson['like_ins']
    json['tag_ins'] = bson['tag_ins']
    json['owner_store_id'] = bson['owner_store_id']
    json['user_like_ids'] = bson['user_like_ids']

    return json

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

        page = int(queries['page'])
        size = int(queries['size'])

        total_documents = db.counters.find_one({'sequenceName': 'cakes'})['seq']

        distances, indices = koclip_index.search(keyword_vector, total_documents)

        faiss_ids = indices[0][page * size: (page + 1) * size].tolist()
        faiss_ids_sorted = [(idx, x) for idx, x in enumerate(faiss_ids)]
        faiss_ids_sorted = sorted(faiss_ids_sorted, key=itemgetter(1))
        
        cake_documents = list(zip(faiss_ids_sorted, list(db.cakes.find({'faiss_id' : {'$in': faiss_ids}}).sort('faiss_id', pymongo.ASCENDING))))
        cake_documents = [to_json(x[1]) for x in sorted(cake_documents, key=itemgetter(0, 0))]

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