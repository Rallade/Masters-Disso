from pymongo import MongoClient
import csv
import ast

client = MongoClient()

db = client.disso

coll = db.simpleBERTcotswaldsMax

with open('simple_embeddings.csv', mode='r', encoding="utf-8") as embeddings:
    reader = csv.DictReader(embeddings, delimiter=";")
    buffer = []
    for i,row in enumerate(reader):
        temp = dict(row)
        temp['Pros_embeddings'] = ast.literal_eval(temp['Pros_embeddings'])
        temp['Cons_embeddings'] = ast.literal_eval(temp['Cons_embeddings'])
        temp['Title_embeddings'] = ast.literal_eval(temp['Title_embeddings'])
        buffer.append(temp)
        if (i % 1000 == 0 and i > 0):
            print(len(buffer), i)
            coll.insert_many(buffer)
            buffer = []