from pymongo import MongoClient
import csv
import numpy as np

from bert_serving.client import BertClient
bc = BertClient()

def fix(db):
    print("fixing missing")

    db_cursor = db.find_missing_pros()
    records = [record for record in db_cursor]
    pros = [r['Pros'] for r in records]

    if pros:
        server_return = bc.encode(pros, show_tokens=True)
        embeddings = server_return[0]
        tokens = server_return[1]
        embeddings = list(map(lambda x,y: x[:len(y)], embeddings, tokens))
        assert len(embeddings) == len(tokens)
        db.attach_listfields_to_records("full_pros_embedding", embeddings, records)
        db.attach_listfields_to_records("pros_tokens", tokens, records) 

    db_cursor = db.find_missing_cons()
    records = [record for record in db_cursor]
    cons = [r['Cons'] for r in records]

    if cons:
        server_return = bc.encode(cons, show_tokens=True)
        embeddings = server_return[0]
        tokens = server_return[1]
        embeddings = list(map(lambda x,y: x[:len(y)], embeddings, tokens))
        assert len(embeddings) == len(tokens)
        db.attach_listfields_to_records("full_cons_embedding", embeddings, records)
        db.attach_listfields_to_records("cons_tokens", tokens, records)


    db_cursor = db.find_missing_titles()
    records = [record for record in db_cursor]
    titles = [r['Product title'] for r in records]

    if titles:
        server_return = bc.encode(titles, show_tokens=True)
        embeddings = server_return[0]
        tokens = server_return[1]
        embeddings = list(map(lambda x,y: x[:len(y)], embeddings, tokens))
        assert len(embeddings) == len(tokens)
        db.attach_listfields_to_records("full_title_embedding", embeddings, records)
        db.attach_listfields_to_records("title_tokens", tokens, records)
