from pymongo import MongoClient

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata

curr_cache_mode = ""
cache = []


def find_pros_with_pooling(pooling_mode):
    return coll.find({pooling_mode + '_pros': {"$exists": True}}, {
                     pooling_mode + '_pros': 1, pooling_mode + '_title': 1, 'href': 1, 'Product title': 1})


def find_pros_with_pooling_cached(pooling_mode):
    global cache
    global curr_cache_mode
    if pooling_mode != curr_cache_mode:
        print("Rebuilding cache")
        cache = list(coll.find({pooling_mode + '_pros': {"$exists": True}}, {
            pooling_mode + '_pros': 1, pooling_mode + '_title': 1, 'href': 1, 'Product title': 1}))
        curr_cache_mode = pooling_mode
    return cache


def find_title_from_link(link):
    return coll.find_one({"href": link})["Product title"]


def find_full_embeddings():
    return coll.find({}, {"_id": 1, "full_pros_embedding": 1, "full_cons_embedding": 1, "full_title_embedding": 1, "pros_token": 1, "cons_tokens": 1, "title_tokens": 1})
