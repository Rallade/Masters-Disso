from pymongo import MongoClient
import pooling
import db_helpers
from multiprocessing import Pool
pool = Pool(6)
import time
from db_upload import attach_listfields_to_records
import stanfordnlp

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata

def paginate(page_size, page_num):
    skips = page_size * page_num
    cursor = coll.find().skip(skips).limit(page_size)

    # Return documents
    return [x for x in cursor]

def create_basic_embeddings():
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

def create_dependency_embeddings():
    data = db_helpers.find_full_embeddings()
    for record in data:
        temp = {}
        pros_tokens, pros_embeddings = remake_tokens(record['pros_tokens'], record['full_pros_embeddings'])
        cons_tokens, cons_embeddings= remake_tokens(record['cons_tokens'], record['full_cons_embeddings'])
        title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embeddings'])
        pros_embeddings = simplify_nested_embeddings(pros_embeddings)
        cons_embeddings = simplify_nested_embeddings(cons_embeddings)
        title_embeddings = simplify_nested_embeddings(title_embeddings)
        
        
def simplify_nested_embeddings(embeddings):
    new = []
    for embedding in embeddings:
        if len(embedding) > 1:
            new.append(pooling.reduce_mean(embedding))
        else:
            new.append(embedding[0])
    return new

def remake_tokens(tokens, embeddings):
    new_tokens = []
    new_embeddings = []
    for i, token in enumerate(tokens):
        if "##" not in token:
            new_tokens.append(token)
            new_embeddings.append([embeddings[i]])
        else:
            new_tokens[-1] += token[2:]
            new_embeddings[-1].append(embeddings[i])
    
    return new_tokens, new_embeddings