from pymongo import MongoClient
import pooling_modes as pooling
from multiprocessing import Pool
pool = Pool(6)
import time
from db_upload import attach_listfields_to_records

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata

def paginate(page_size, page_num):
    skips = page_size * page_num
    cursor = coll.find().skip(skips).limit(page_size)

    # Return documents
    return [x for x in cursor]


batch_size = 1500
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
    pro_means = list(pool.map(pooling.reduce_mean,embeddings))
    pro_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
    pro_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

    embeddings = []
    for record in records:
        try:
            embeddings.append(record['full_cons_embedding'])
        except:
            embeddings.append(None)
    con_means = list(pool.map(pooling.reduce_mean,embeddings))
    con_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
    con_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

    embeddings = []
    for record in records:
        try:
            embeddings.append(record['full_title_embedding'])
        except:
            embeddings.append(None)
    title_means = list(pool.map(pooling.reduce_mean,embeddings))
    title_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
    title_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))


    attach_listfields_to_records("mean_pooling_pros", pro_means, records, coll)
    attach_listfields_to_records("max_pooling_single_pros", pro_maxes_s, records, coll)
    attach_listfields_to_records("max_pooling_total_pros", pro_maxes_t, records, coll)
    attach_listfields_to_records("mean_pooling_cons", con_means, records, coll)
    attach_listfields_to_records("max_pooling_single_cons", con_maxes_s, records, coll)
    attach_listfields_to_records("max_pooling_total_cons", con_maxes_t, records, coll)
    attach_listfields_to_records("mean_pooling_title", title_means, records, coll)
    attach_listfields_to_records("max_pooling_single_title", title_maxes_s, records, coll)
    attach_listfields_to_records("max_pooling_total_title", title_maxes_t, records, coll)

    print("Records done:", i * batch_size)

    if not records:
        break
t1 = time.time()
print(t1-t0)
