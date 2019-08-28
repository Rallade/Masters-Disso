from bert_serving.client import BertClient
import numpy as np
bc = BertClient()


number_of_processors = 1  # no. of GPUs
batch_size = 256

def create(db, number_of_processors, batch_size):
    print("Starting Pros")
    db_cursor = db.find()
    full = list(db_cursor)
    full = sorted(full, key=lambda x: len(x['Pros']))
    batch = []
    for i, record in enumerate(full):
        batch.append(record)
        if i % (number_of_processors*batch_size) == (number_of_processors*batch_size)-1:
            curr_batch = list(map(lambda x: x['Pros'], batch))
            # remove all empty strings
            try:
                idx_of_last_empty = len(curr_batch) - curr_batch[::-1].index("") - 1
            except:
                idx_of_last_empty = -1
            curr_batch = curr_batch[idx_of_last_empty + 1:]
            if curr_batch:
                server_return = bc.encode(curr_batch, show_tokens=True)
                pros_embeddings = server_return[0]
                tokens = server_return[1]
                pros_embeddings = list(map(lambda x,y: x[:len(y)], pros_embeddings, tokens))
                batch = batch[idx_of_last_empty + 1:]
                assert len(pros_embeddings) == len(batch) == len(tokens)
                db.attach_listfields_to_records("full_pros_embedding", pros_embeddings, batch)
                db.attach_listfields_to_records("pros_tokens", tokens, batch)
                print("Percentage till completion:", str(i*100/len(full)) + "%")
            batch = []

    print('Starting Cons')

    full = sorted(full, key=lambda x: len(x['Cons']))
    for i, record in enumerate(full):
        batch.append(record)
        if i % (number_of_processors*batch_size) == (number_of_processors*batch_size)-1:
            curr_batch = list(map(lambda x: x['Cons'], batch))
            # remove all empty strings
            try:
                idx_of_last_empty = len(curr_batch) - curr_batch[::-1].index("") - 1
            except:
                idx_of_last_empty = -1
            curr_batch = curr_batch[idx_of_last_empty + 1:]
            if curr_batch:
                server_return = bc.encode(curr_batch, show_tokens=True)
                cons_embeddings = server_return[0]
                tokens = server_return[1]
                cons_embeddings = list(map(lambda x,y: x[:len(y)], cons_embeddings, tokens))
                batch = batch[idx_of_last_empty + 1:]
                assert len(cons_embeddings) == len(batch) == len(tokens)
                db.attach_listfields_to_records("full_cons_embedding", cons_embeddings, batch)
                db.attach_listfields_to_records("cons_tokens", tokens, batch)
                print("Percentage till completion:", str(i*100/len(full)) + "%")
            batch = []



    print("Starting titles")

    full = sorted(full, key=lambda x: len(x['Product title']))
    for i, record in enumerate(full):
        batch.append(record)
        if i % (number_of_processors*batch_size) == (number_of_processors*batch_size)-1:
            curr_batch = list(map(lambda x: x['Product title'], batch))
            # remove all empty strings
            try:
                idx_of_last_empty = len(curr_batch) - curr_batch[::-1].index("") - 1
            except:
                idx_of_last_empty = -1
            curr_batch = curr_batch[idx_of_last_empty + 1:]
            if curr_batch:
                server_return = bc.encode(curr_batch, show_tokens=True)
                cons_embeddings = server_return[0]
                tokens = server_return[1]
                cons_embeddings = list(map(lambda x,y: x[:len(y)], cons_embeddings, tokens))
                batch = batch[idx_of_last_empty + 1:]
                assert len(cons_embeddings) == len(batch) == len(tokens)
                db.attach_listfields_to_records("full_title_embedding", cons_embeddings, batch)
                db.attach_listfields_to_records("title_tokens", tokens, batch)
                print("Percentage till completion:", str(i*100/len(full)) + "%")
            batch = []