from pymongo import MongoClient
import csv
import numpy as np

class DB_helpers:

    def __init__(self, db_name):
        self.client = MongoClient()
        self.db = self.client.disso
        self.coll = self.db[db_name]
        self.curr_cache_mode = ""

    def find_pros_with_pooling(self, pooling_mode):
        return self.coll.find({pooling_mode + '_pros': {"$exists": True}}, {
                            pooling_mode + '_pros': 1, pooling_mode + '_title': 1, 'href': 1, 'Product title': 1, 'Pros':1})
    
    
    def find_pros_with_pooling_cached(self, pooling_mode):
        if pooling_mode != self.curr_cache_mode:
            print("Rebuilding cache")
            self.cache = list(self.coll.find({pooling_mode + '_pros': {"$exists": True}}, {
                pooling_mode + '_pros': 1, pooling_mode + '_title': 1, 'href': 1, 'Product title': 1, "Pros": 1}))
            self.curr_cache_mode = pooling_mode
        return self.cache
    
    
    def find_title_from_link(self, link):
        return self.coll.find_one({"href": link})["Product title"]
    
    
    def find_full_embeddings(self):
        return self.coll.find({}, {"_id": 1, "full_pros_embedding": 1, "full_cons_embedding": 1, "full_title_embedding": 1, "pros_tokens": 1, "cons_tokens": 1, "title_tokens": 1})

    def paginate(self, page_size, page_num):
        skips = page_size * page_num
        cursor = self.coll.find().skip(skips).limit(page_size)
    
        # Return documents
        return [x for x in cursor]

    def upload_csv(self, filename):
        with open(filename, encoding="utf-8") as nd_file:
            reader = csv.DictReader(nd_file)
            buffer = []
            for i, row in enumerate(reader):
                buffer.append(row)
                if i % 10000 == 0:
                    self.coll.insert_many(buffer)
                    buffer = []
    
    
    def attach_listfields_to_records(self, field_name, new, old):
        for i, record in enumerate(old):
            field = new[i]
            if(isinstance(new[i], np.ndarray)):
                field = new[i].tolist()
            if field is not None:
                self.coll.update_one(
                    {"_id": record['_id']},
                    {
                        "$set": {
                            field_name: field
                        }
                    }
    
            )
    
    
    def update_one(self, field_name, data, record):
        field = data
        if isinstance(field, list):
            field = np.array(field)
        if isinstance(field, np.ndarray):
            field = field.tolist()
        if field is not None:
            self.coll.update_one(
                    {"_id": record['_id']},
                    {
                        "$set": {
                            field_name: field
                        }
                    }
            )
    
    def drop(self):
        self.coll.drop()

    def find(self):
        return self.coll.find()
    
    def find_missing_pros(self):
        return self.coll.find({"Pros": {"$exists": True, "$ne": ""}, "pros_tokens": {"$exists": False}})

    def find_missing_cons(self):
        return self.coll.find({"Cons": {"$exists": True, "$ne": ""}, "cons_tokens": {"$exists": False}})

    def find_missing_titles(self):
        return self.coll.find({"Product title": {"$exists": True, "$ne": ""}, "title_tokens": {"$exists": False}})