from pymongo import MongoClient

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata


data = coll.find()
titles = set()

with open('training_data_titles_reviews.txt', mode='w', encoding="utf-8") as training_data:
    for datum in data:
        titles.add(datum["Product title"])
        try:
            titles.add(datum["Pros"])
        except:
            pass
        try:
            titles.add(datum["Cons"])
        except:
            pass
    for title in titles:
        training_data.write(title+"\n\n")
