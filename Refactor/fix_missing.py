from pymongo import MongoClient
import csv
import numpy as np
import db_upload

from bert_serving.client import BertClient
bc = BertClient()

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata

db_cursor = coll.find({"Pros": {"$exists": True, "$ne": ""}, "pros_tokens": {"$exists": False}})
records = [record for record in db_cursor]
pros = [r['Pros'] for r in records]
try:
    server_return = bc.encode(pros, show_tokens=True)
    embeddings = server_return[0]
    tokens = server_return[1]
    embeddings = list(map(lambda x,y: x[:len(y)], embeddings, tokens))
    assert len(embeddings) == len(tokens)
    db_upload.attach_listfields_to_records("full_pros_embedding", embeddings, records, coll)
    db_upload.attach_listfields_to_records("pros_tokens", tokens, records, coll)
except:
    pass

db_cursor = coll.find({"Cons": {"$exists": True, "$ne": ""}, "cons_tokens": {"$exists": False}})
records = [record for record in db_cursor]
cons = [r['Cons'] for r in records]
try:
    server_return = bc.encode(cons, show_tokens=True)
    embeddings = server_return[0]
    tokens = server_return[1]
    embeddings = list(map(lambda x,y: x[:len(y)], embeddings, tokens))
    assert len(embeddings) == len(tokens)
    db_upload.attach_listfields_to_records("full_cons_embedding", embeddings, records, coll)
    db_upload.attach_listfields_to_records("cons_tokens", tokens, records, coll)
except:
    pass

db_cursor = coll.find({"Product title": {"$exists": True, "$ne": ""}, "title_tokens": {"$exists": False}})
records = [record for record in db_cursor]
titles = [r['Product title'] for r in records]
try:
    server_return = bc.encode(titles, show_tokens=True)
    embeddings = server_return[0]
    tokens = server_return[1]
    embeddings = list(map(lambda x,y: x[:len(y)], embeddings, tokens))
    assert len(embeddings) == len(tokens)
    db_upload.attach_listfields_to_records("full_title_embedding", embeddings, records, coll)
    db_upload.attach_listfields_to_records("title_tokens", tokens, records, coll)
except:
    pass