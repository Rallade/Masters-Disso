from pymongo import MongoClient
import csv
import numpy as np

class DB_helpers:
    """
    Abstraction layer for the database.
    Any interaction with the database must occur here
    for consistency and modularity.

    Attributes
    ----------

    db_name: str
        Name of the collection

    Methods
    -------
    
    find()
        returns an iterable to every item in the database
    
    upload_csv()
        takes a CSV file and uses its data to build a database
    
    find_missing_[field name]()
        returns an iterable containing all the records in that
        field that do not have embeddings
    
    upload(entries: iterable of dict)
        each dict must contain an "_id" field (in the case of MongoDB)
        insert all the 

    drop()
        drops the collection
    
    find_full_embeddings()
        returns  an iterable with the full embeddings per record
    
    find_pros_with_pooling( pooling_mode : str)
        returns an iterable with pros review embedding created using
        a specific pooling mode
    
    find_title_from_link(link : str)
        returns the string containing the title of a product, given a link

    paginate(page_size: uint , page_num: uint)
        paginates the find() command
        starting at page_num
    
    update_one(field_name: str, data: anything, record: dict)
        given a database object (record) add a field to that record
        with the name field_name with the given data.

    attach_listfields_to_records(self, field_name: str, new: list, old: list of dict)
        given a field name, attach data in new into records in old into
        that field name

    """


    def __init__(self, db_name):
        self.client = MongoClient()
        self.db = self.client.disso
        self.coll = self.db[db_name]
        self.curr_cache_mode = ""

    def find_pros_with_pooling(self, pooling_mode: str):
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

    def upload(self, entries):
        return self.coll.insert_many(entries).inserted_ids