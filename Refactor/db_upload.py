import csv
import pymongo
import numpy as np


def upload(filename, collection):
    with open("no_duplicates.csv", encoding="utf-8") as nd_file:
        reader = csv.DictReader(nd_file)
        buffer = []
        for i, row in enumerate(reader):
            buffer.append(row)
            if i % 10000 == 0:
                collection.insert_many(buffer)
                buffer = []


def attach_listfields_to_records(field_name, new, old, collection):
    for i, record in enumerate(old):
        field = new[i]
        if(isinstance(new[i], np.ndarray)):
            field = new[i].tolist()
        if field is not None:
            collection.update_one(
                {"_id": record['_id']},
                {
                    "$set": {
                        field_name: field
                    }
                }

        )


def update_one(field_name, data, record, collection):
    field = data
    if isinstance(field, list):
        field = np.array(field)
    if isinstance(field, np.ndarray):
        field = field.tolist()
    if field is not None:
        collection.update_one(
                {"_id": record['_id']},
                {
                    "$set": {
                        field_name: field
                    }
                }
        )
