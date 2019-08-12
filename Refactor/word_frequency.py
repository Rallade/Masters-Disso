from pymongo import MongoClient
client = MongoClient()
db = client.disso
coll = db.cotswaldsdata
from add_pooling_to_DB import remake_tokens, simplify_nested_embeddings

cursor = coll.find()


count = {}

for record in cursor:
    try:
        pros_tokens, pros_embeddings = remake_tokens(record['pros_tokens'], record['full_pros_embedding'])
        for token in pros_tokens:
            try:
                count[token] += 1
            except KeyError:
                count[token] = 1
    except:
        pass
    try:
        cons_tokens, cons_embeddings = remake_tokens(record['cons_tokens'], record['full_cons_embedding'])
        for token in cons_tokens:
            try:
                count[token] += 1
            except KeyError:
                count[token] = 1
    except:
        pass
    title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embedding'])
    for token in title_tokens:
        try:
            count[token] += 1
        except KeyError:
            count[token] = 1

sorted_counts = sorted(count.items(), key=lambda x: x[1], reverse=True)
print(sorted_counts[:200])