from pymongo import MongoClient
import pooling_modes as pooling
import numpy as np
from multiprocessing import Pool
pool = Pool(6)
import time
from db_upload import attach_listfields_to_records

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata_copy

def paginate(page_size, page_num):
    skips = page_size * page_num
    cursor = coll.find().skip(skips).limit(page_size)

    # Return documents
    return [x for x in cursor]


batch_size = 3000
i = 0
t0 = time.time()
while True:
    records = paginate(batch_size, i)
    i += 1
    embeddings = []
    for record in records:
        try:
            embeddings.append(record['full_pros_embedding'])
        except:
            embeddings.append(None)
    pro_means = list(pool.map(pooling.reduce_max,embeddings))
    embeddings = []
    for record in records:
        try:
            embeddings.append(record['full_cons_embedding'])
        except:
            embeddings.append(None)
    con_means = list(pool.map(pooling.reduce_max,embeddings))
    attach_listfields_to_records("mean_pooling_pros", pro_means, records, coll)
    attach_listfields_to_records("mean_pooling_cons", con_means, records, coll)
    if not records:
        break
t1 = time.time()
print(t1-t0)
