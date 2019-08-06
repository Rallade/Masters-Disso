from pymongo import MongoClient

client = MongoClient()
db = client.disso
coll = db.simpleBERTcotswaldsMax


data = coll.find()
titles = set()

with open('training_data.txt', mode='w', encoding="utf-8") as training_data:
    for datum in data:
        titles.add(datum["Title"])
    for title in titles:
        training_data.write(title+"\n\n")