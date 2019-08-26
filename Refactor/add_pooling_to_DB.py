import pooling
from db_helpers import DB_helpers
from multiprocessing import Pool
pool = Pool(6)
import time
import stanfordnlp
import re
import nltk

db = DB_helpers("screwfixdata")

def create_basic_embeddings():
    batch_size = 1500
    i = 0
    t0 = time.time()
    while True:
        records = db.paginate(batch_size, i)
        i += 1
        
        embeddings = []
        for record in records:
            try:
                embeddings.append(record['full_pros_embedding'])
            except:
                embeddings.append(None)
        pro_means = list(pool.map(pooling.reduce_mean,embeddings))
        pro_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        pro_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

        embeddings = []
        for record in records:
            try:
                embeddings.append(record['full_cons_embedding'])
            except:
                embeddings.append(None)
        con_means = list(pool.map(pooling.reduce_mean,embeddings))
        con_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        con_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

        embeddings = []
        for record in records:
            try:
                embeddings.append(record['full_title_embedding'])
            except:
                embeddings.append(None)
        title_means = list(pool.map(pooling.reduce_mean,embeddings))
        title_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        title_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))


        db.attach_listfields_to_records("mean_pooling_pros", pro_means, records)
        db.attach_listfields_to_records("max_pooling_single_pros", pro_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_total_pros", pro_maxes_t, records)
        db.attach_listfields_to_records("mean_pooling_cons", con_means, records)
        db.attach_listfields_to_records("max_pooling_single_cons", con_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_total_cons", con_maxes_t, records)
        db.attach_listfields_to_records("mean_pooling_title", title_means, records)
        db.attach_listfields_to_records("max_pooling_single_title", title_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_total_title", title_maxes_t, records)

        print("Records done:", i * batch_size)

        if not records:
            break
    t1 = time.time()
    print(t1-t0)

def create_basic_embeddings_appended_title():
    batch_size = 1500
    i = 33000
    t0 = time.time()
    while True:
        records = db.paginate(batch_size, i)
        i += 1
        
        embeddings = []
        for record in records:
            temp = []
            try:
                temp.extend(record['full_title_embedding'])
                temp.extend(record['full_pros_embedding'][1:]) #remove start token
                embeddings.append(temp)
            except KeyError:
                embeddings.append(None)
        pro_means = list(pool.map(pooling.reduce_mean,embeddings))
        pro_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        pro_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

        embeddings = []
        for record in records:
            temp = []
            try:
                temp.extend(record['full_title_embedding'])
                temp.extend(record['full_cons_embedding'][1:]) #remove start token
                embeddings.append(temp)
            except:
                embeddings.append(None)
        con_means = list(pool.map(pooling.reduce_mean,embeddings))
        con_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        con_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))
        
        embeddings = []
        for record in records:
            try:
                embeddings.append(record['full_title_embedding'])
            except:
                embeddings.append(None)
        title_means = list(pool.map(pooling.reduce_mean,embeddings))
        title_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        title_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))


        db.attach_listfields_to_records("mean_pooling_pros", pro_means, records)
        db.attach_listfields_to_records("max_pooling_single_pros", pro_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_total_pros", pro_maxes_t, records)
        db.attach_listfields_to_records("mean_pooling_cons", con_means, records)
        db.attach_listfields_to_records("max_pooling_single_cons", con_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_total_cons", con_maxes_t, records)
        db.attach_listfields_to_records("mean_pooling_title", title_means, records)
        db.attach_listfields_to_records("max_pooling_single_title", title_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_total_title", title_maxes_t, records)

        print("Records done:", i * batch_size)

        if not records:
            break
    t1 = time.time()
    print(t1-t0)

def fix_tokens(tokens, embeddings):
    new_embeddings = embeddings
    new_tokens = tokens
    new_tokens = [token if token != "[UNK]" else "'" for token in new_tokens]
    new_tokens = [token if token != "[SEP]" else "." for token in new_tokens]
    for i, token in enumerate(new_tokens):
        print(new_tokens, new_tokens[i])
        print(tokens, re.search(r"^\d?\d\D+", new_tokens[i]))
        if new_tokens[i] == "[UNK]":
            new_tokens[i] = "'"
        elif new_tokens[i] == "[SEP]":
            new_tokens[i] = "."
        elif new_tokens[i] == "!" and new_tokens[i+1] == "!":
            new_tokens[i] = "!!"
            new_tokens.pop(i+1)
            new_embeddings.pop(i+1)
            i-=1
        elif new_tokens[i] == "its":
            new_tokens[i] = "it"
            new_tokens.insert(i+1,"s")
            new_embeddings.insert(i+1,new_embeddings[i])
        elif new_tokens[i] == "":
            try:
                assert len(new_tokens) == len(new_embeddings)
            except:
                print(len(new_embeddings), len(new_tokens))
                print(new_tokens, new_tokens[i])
                raise AssertionError
            new_tokens.pop(i)
            new_embeddings.pop(i)
        elif re.search(r"^\d*\d\D+", new_tokens[i]):
            x = re.search(r"\D", new_tokens[i])
            new_tokens.insert(i+1,new_tokens[i][x.span()[1]-1:])
            new_tokens[i] = new_tokens[i][:x.span()[1]-1]
            new_embeddings.insert(i+1,new_embeddings[i])
        elif re.search(r"^£\d+", new_tokens[i]):
            x = re.search(r"\d", new_tokens[i])
            new_tokens.insert(i+1,new_tokens[i][x.span()[1]-1:])
            new_tokens[i] = new_tokens[i][:x.span()[1]-1]
            new_embeddings.insert(i+1,new_embeddings[i])
    return new_tokens, new_embeddings


def create_nltk_pos_embeddings_appended_title():
    data = db.find_full_embeddings()
    k = 0
    for record in data:
        try:
            title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embedding'])
            title_embeddings = simplify_nested_embeddings(title_embeddings)
            title_tokens = ["." if token == "[SEP]" else token for token in title_tokens]
            title = " ".join(title_tokens[1:-1])
            pos_title = nltk.pos_tag(title_tokens[1:-1])
            #print(list(zip(title_tokens[1:-1], pos_title)))
            new_title_embeddings = []
            new_title_tokens = []
            for i, pos in enumerate(pos_title):
                if ('JJ' in pos[1] or 'NN' in pos[1]) and  ("|" not in pos[0]):
                    new_title_embeddings.append(title_embeddings[i+1]) #account for [CLS] token
                    new_title_tokens.append(pos[0])
                if pos[0] != title_tokens[i+1]:
                    [print(pc) for pc in pos_title]
                    print(title_tokens)
                    print(title)
                    print(title_tokens[i+1])
                    raise ValueError
            if len(new_title_tokens) == 0:
                new_title_tokens = title_tokens
                new_title_embeddings = title_embeddings

            pros_tokens, pros_embeddings = remake_tokens(record['pros_tokens'], record['full_pros_embedding'])
            pros_embeddings = simplify_nested_embeddings(pros_embeddings)
            pros_tokens = ["." if token == "[SEP]" else token for token in pros_tokens]
            pros = " ".join(pros_tokens[1:-1])
            pos_pros = nltk.pos_tag(pros_tokens[1:-1])
            #print(list(zip(pros_tokens[1:-1], pos_pros)))
            new_pros_embeddings = []
            new_pros_tokens = []
            for i, pos in enumerate(pos_pros):
                if (pos[1] == 'JJ' or 'VB' in pos[1] or pos[1] == 'NN') and  ("|" not in pos[0]):
                    new_pros_embeddings.append(pros_embeddings[i+1]) #account for [CLS] token
                    new_pros_tokens.append(pos[0])
                if pos[0] != pros_tokens[i+1]:
                    [print(pc) for pc in pos_pros]
                    print(pros_tokens)
                    print(pros)
                    print(pros_tokens[i+1])
                    raise ValueError
            if new_pros_embeddings or new_title_embeddings:
                new_title_embeddings.extend(new_pros_embeddings)
                new_title_tokens.extend(new_pros_tokens)
                db.update_one("pos_filtered_pros_embedding", new_title_embeddings, record)
                db.update_one("pos_filtered_pros_tokens", new_title_tokens, record)
        except KeyError:
            pass
        
        try:
            title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embedding'])
            title_embeddings = simplify_nested_embeddings(title_embeddings)
            title_tokens = ["." if token == "[SEP]" else token for token in title_tokens]
            title = " ".join(title_tokens[1:-1])
            pos_title = nltk.pos_tag(title_tokens[1:-1])
            #print(list(zip(title_tokens[1:-1], pos_title)))
            new_title_embeddings = []
            new_title_tokens = []
            for i, pos in enumerate(pos_title):
                if ('JJ' in pos[1] or 'NN' in pos[1]) and  ("|" not in pos[0]):
                    new_title_embeddings.append(title_embeddings[i+1]) #account for [CLS] token
                    new_title_tokens.append(pos[0])
                if pos[0] != title_tokens[i+1]:
                    [print(pc) for pc in pos_title]
                    print(title_tokens)
                    print(title)
                    print(title_tokens[i+1])
                    raise ValueError
            if len(new_title_tokens) == 0:
                new_title_tokens = title_tokens
                new_title_embeddings = title_embeddings
            cons_tokens, cons_embeddings = remake_tokens(record['cons_tokens'], record['full_cons_embedding'])
            cons_embeddings = simplify_nested_embeddings(cons_embeddings)
            cons_tokens = ["." if token == "[SEP]" else token for token in cons_tokens]
            cons = " ".join(cons_tokens[1:-1])
            pos_cons = nltk.pos_tag(cons_tokens[1:-1])
            #print(list(zip(cons_tokens[1:-1], pos_cons)))
            new_cons_embeddings = []
            new_cons_tokens = []
            for i, pos in enumerate(pos_cons):
                if (pos[1] == 'JJ' or 'VB' in pos[1] or pos[1] == 'NN') and  ("|" not in pos[0]):
                    new_cons_embeddings.append(cons_embeddings[i+1]) #account for [CLS] token
                    new_cons_tokens.append(pos[0])
                if pos[0] != cons_tokens[i+1]:
                    [print(pc) for pc in pos_cons]
                    print(cons_tokens)
                    print(cons)
                    print(cons_tokens[i+1])
                    raise ValueError
            if new_cons_embeddings or new_title_embeddings:
                new_title_embeddings.extend(new_cons_embeddings)
                new_title_tokens.extend(new_cons_tokens)
                db.update_one("pos_filtered_cons_embedding", new_title_embeddings, record)
                db.update_one("pos_filtered_cons_tokens", new_title_tokens, record)
        except KeyError:
            pass
        
        try:
            title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embedding'])
            title_embeddings = simplify_nested_embeddings(title_embeddings)
            title_tokens = ["." if token == "[SEP]" else token for token in title_tokens]
            title = " ".join(title_tokens[1:-1])
            pos_title = nltk.pos_tag(title_tokens[1:-1])
            #print(list(zip(title_tokens[1:-1], pos_title)))
            new_title_embeddings = []
            new_title_tokens = []
            for i, pos in enumerate(pos_title):
                if ('JJ' in pos[1] or 'NN' in pos[1]) and  ("|" not in pos[0]):
                    new_title_embeddings.append(title_embeddings[i+1]) #account for [CLS] token
                    new_title_tokens.append(pos[0])
                if pos[0] != title_tokens[i+1]:
                    [print(pc) for pc in pos_title]
                    print(title_tokens)
                    print(title)
                    print(title_tokens[i+1])
                    raise ValueError
            if len(new_title_tokens) == 0:
                new_title_tokens = title_tokens
                new_title_embeddings = title_embeddings
            if new_title_embeddings:
                db.update_one("pos_filtered_title_embedding", new_title_embeddings, record)
                db.update_one("pos_filtered_title_tokens", new_title_tokens, record)
        except KeyError:
            pass
        k += 1
        if k%100 == 0:
            print("Completed records:", k)
    pool_pos_embeddings()

def create_nltk_pos_embeddings():
    data = db.find_full_embeddings()
    k = 0
    for record in data:
        try:
            pros_tokens, pros_embeddings = remake_tokens(record['pros_tokens'], record['full_pros_embedding'])
            pros_embeddings = simplify_nested_embeddings(pros_embeddings)
            pros_tokens = ["." if token == "[SEP]" else token for token in pros_tokens]
            pros = " ".join(pros_tokens[1:-1])
            pos_pros = nltk.pos_tag(pros_tokens[1:-1])
            #print(list(zip(pros_tokens[1:-1], pos_pros)))
            new_pros_embeddings = []
            new_pros_tokens = []
            for i, pos in enumerate(pos_pros):
                if (pos[1] == 'JJ' or 'VB' in pos[1] or pos[1] == 'NN') and  ("|" not in pos[0]):
                    new_pros_embeddings.append(pros_embeddings[i+1]) #account for [CLS] token
                    new_pros_tokens.append(pos[0])
                if pos[0] != pros_tokens[i+1]:
                    [print(pc) for pc in pos_pros]
                    print(pros_tokens)
                    print(pros)
                    print(pros_tokens[i+1])
                    raise ValueError
            if new_pros_embeddings:
                db.update_one("pos_filtered_pros_embedding", new_pros_embeddings, record)
                db.update_one("pos_filtered_pros_tokens", new_pros_tokens, record)
        except KeyError:
            pass
        try:
            cons_tokens, cons_embeddings = remake_tokens(record['cons_tokens'], record['full_cons_embedding'])
            cons_embeddings = simplify_nested_embeddings(cons_embeddings)
            cons_tokens = ["." if token == "[SEP]" else token for token in cons_tokens]
            cons = " ".join(cons_tokens[1:-1])
            pos_cons = nltk.pos_tag(cons_tokens[1:-1])
            #print(list(zip(cons_tokens[1:-1], pos_cons)))
            new_cons_embeddings = []
            new_cons_tokens = []
            for i, pos in enumerate(pos_cons):
                if (pos[1] == 'JJ' or 'VB' in pos[1] or pos[1] == 'NN') and  ("|" not in pos[0]):
                    new_cons_embeddings.append(cons_embeddings[i+1]) #account for [CLS] token
                    new_cons_tokens.append(pos[0])
                if pos[0] != cons_tokens[i+1]:
                    [print(pc) for pc in pos_cons]
                    print(cons_tokens)
                    print(cons)
                    print(cons_tokens[i+1])
                    raise ValueError
            if new_cons_embeddings:
                db.update_one("pos_filtered_cons_embedding", new_cons_embeddings, record)
                db.update_one("pos_filtered_cons_tokens", new_cons_tokens, record)
        except KeyError:
            pass
        try:
            title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embedding'])
            title_embeddings = simplify_nested_embeddings(title_embeddings)
            title_tokens = ["." if token == "[SEP]" else token for token in title_tokens]
            title = " ".join(title_tokens[1:-1])
            pos_title = nltk.pos_tag(title_tokens[1:-1])
            #print(list(zip(title_tokens[1:-1], pos_title)))
            new_title_embeddings = []
            new_title_tokens = []
            for i, pos in enumerate(pos_title):
                if ('JJ' in pos[1] or 'NN' in pos[1]) and  ("|" not in pos[0]):
                    new_title_embeddings.append(title_embeddings[i+1]) #account for [CLS] token
                    new_title_tokens.append(pos[0])
                if pos[0] != title_tokens[i+1]:
                    [print(pc) for pc in pos_title]
                    print(title_tokens)
                    print(title)
                    print(title_tokens[i+1])
                    raise ValueError
            if len(new_title_tokens) == 0:
                new_title_tokens = title_tokens
                new_title_embeddings = title_embeddings
            if new_title_embeddings:
                db.update_one("pos_filtered_title_embedding", new_title_embeddings, record)
                db.update_one("pos_filtered_title_tokens", new_title_tokens, record)
        except KeyError:
            pass
        k += 1
        if k%100 == 0:
            print("Completed records:", k)
    pool_pos_embeddings()
        
def pool_pos_embeddings():
    batch_size = 1500
    i = 0
    t0 = time.time()
    while True:
        records = db.paginate(batch_size, i)
        i += 1
        
        embeddings = []
        for record in records:
            try:
                embeddings.append(record['pos_filtered_pros_embedding'])
            except:
                embeddings.append(None)
        pro_means = list(pool.map(pooling.reduce_mean,embeddings))
        pro_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        pro_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

        embeddings = []
        for record in records:
            try:
                embeddings.append(record['pos_filtered_cons_embedding'])
            except:
                embeddings.append(None)
        con_means = list(pool.map(pooling.reduce_mean,embeddings))
        con_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        con_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))

        embeddings = []
        for record in records:
            try:
                embeddings.append(record['pos_filtered_title_embedding'])
            except:
                embeddings.append(None)
        title_means = list(pool.map(pooling.reduce_mean,embeddings))
        title_maxes_s = list(pool.map(pooling.reduce_max_single,embeddings))
        title_maxes_t = list(pool.map(pooling.reduce_max_total,embeddings))


        db.attach_listfields_to_records("mean_pooling_pos_filtered_pros", pro_means, records)
        db.attach_listfields_to_records("max_pooling_pos_filtered_single_pros", pro_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_pos_filtered_total_pros", pro_maxes_t, records)
        db.attach_listfields_to_records("mean_pooling_pos_filtered_cons", con_means, records)
        db.attach_listfields_to_records("max_pooling_pos_filtered_single_cons", con_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_pos_filtered_total_cons", con_maxes_t, records)
        db.attach_listfields_to_records("mean_pooling_pos_filtered_title", title_means, records)
        db.attach_listfields_to_records("max_pooling_pos_filtered_single_title", title_maxes_s, records)
        db.attach_listfields_to_records("max_pooling_pos_filtered_total_title", title_maxes_t, records)

        print("Records done:", i * batch_size)

        if not records:
            break
    t1 = time.time()
    print(t1-t0)


def create_dependency_embeddings():
    data = db.find_full_embeddings()
    nlp = stanfordnlp.Pipeline()

    for record in data:
        try:
            pros_tokens, pros_embeddings = remake_tokens(record['pros_tokens'], record['full_pros_embedding'])
            pros_embeddings = simplify_nested_embeddings(pros_embeddings)
            pros_tokens, pros_embeddings = fix_tokens(pros_tokens, pros_embeddings)
            pros = " ".join(pros_tokens[1:-1])
            pos_pros = [word for sent in nlp(pros).sentences for word in sent.words]
            #print(list(zip(pros_tokens[1:-1], pos_pros)))
            new_pros_embeddings = []
            for i, pos in enumerate(pos_pros):
                if pos.text != pros_tokens[i+1]:
                    pros_tokens, pros_embeddings = fix_tokens(pros_tokens, pros_embeddings)
                    [print(pp) for pp in pos_pros]
                    print(pros_tokens)
                    print(pros)
                    print(pos.text, pros_tokens[i+1])
                    raise ValueError
                if pos.upos == 'ADJ' or pos.upos == 'VERB' or pos.upos == 'NOUN':
                    new_pros_embeddings.append(pros_embeddings[i+1]) #account for [CLS] token
            if new_pros_embeddings:
                db.update_one("pos_filtered_pros_embedding", new_pros_embeddings, record)
        except KeyError:
            pass
        try:
            cons_tokens, cons_embeddings = remake_tokens(record['cons_tokens'], record['full_cons_embedding'])
            cons_embeddings = simplify_nested_embeddings(cons_embeddings)
            cons_tokens, cons_embeddings = fix_tokens(cons_tokens, cons_embeddings)
            cons = " ".join(cons_tokens[1:-1])
            pos_cons = [word for sent in nlp(cons).sentences for word in sent.words]
            new_cons_embeddings = []
            for i, pos in enumerate(pos_cons):
                if pos.text != cons_tokens[i+1]:
                    [print(pc) for pc in pos_cons]
                    print(cons_tokens)
                    print(cons)
                    print(pos.text, cons_tokens[i+1])
                    raise ValueError
                if pos.upos == 'ADJ' or pos.upos == 'VERB' or pos.upos == 'NOUN':
                    new_cons_embeddings.append(cons_embeddings[i+1]) #account for [CLS] token
            if new_cons_embeddings:
                db.update_one("pos_filtered_cons_embedding", new_cons_embeddings, record)
        except KeyError:
            pass
        try:
            title_tokens, title_embeddings = remake_tokens(record['title_tokens'], record['full_title_embedding'])
            title_embeddings = simplify_nested_embeddings(title_embeddings)
            title = " ".join(title_tokens[1:-1])
            pos_title = [word.upos for sent in nlp(title).sentences for word in sent.words]
            #print(list(zip(title_tokens[1:-1], pos_title)))
            new_title_embeddings = []
            for i, pos in enumerate(pos_title):
                if pos == 'ADJ' or pos == 'NOUN':
                    new_title_embeddings.append(title_embeddings[i+1]) #account for [CLS] token
            if new_title_embeddings:
                db.update_one("pos_filtered_title_embedding", new_title_embeddings, record)
        except KeyError:
            pass
        
        
def simplify_nested_embeddings(embeddings):
    new = []
    for embedding in embeddings:
        if len(embedding) > 1:
            new.append(pooling.reduce_mean(embedding))
        else:
            new.append(embedding[0])
    return new

def remake_tokens(tokens, embeddings):
    new_tokens = []
    new_embeddings = []
    for i, token in enumerate(tokens):
        if "##" not in token:
            new_tokens.append(token)
            new_embeddings.append([embeddings[i]])
        else:
            new_tokens[-1] += token[2:]
            new_embeddings[-1].append(embeddings[i])
    
    return new_tokens, new_embeddings