import numpy as np
from bert_serving.client import BertClient
from collections import OrderedDict
from functools import reduce
import pooling
import db_helpers
import heapq
from multiprocessing import Pool
pool = Pool(6)
from itertools import repeat

pooling_modes = ["mean_pooling", "max_pooling_single", "max_pooling_total",
                 "mean_pooling_pos_filtered", "max_pooling_pos_filtered_single", "max_pooling_pos_filtered_total"]

def dist(a, b):
    if b is not None:
        return 1-a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        return 1

class Search:
    def __init__(self, db_name, pooling_mode, cache=True):
        self.bc = BertClient()
        self.db = db_helpers.DB_helpers(db_name)
        if cache:
            self.db.find_pros_with_pooling_cached(pooling_mode)
        self.pooling_mode = pooling_mode
        self.embed_cache = []
        self.data_cache = []
        self.cache_mode = ""

    def query_many(self, phrases, top_cut=10, decay_factor=1, results=10):
        res = self.bc.encode(phrases, show_tokens=True)
        res = list(zip(res[0],repeat(self.pooling_mode), res[1]))
        queries = list(pool.starmap(pooling.pool, res))
        queries = list(zip(zip(queries, [r[2] for r in res]), repeat(top_cut), repeat(decay_factor), repeat(results)))
        print(queries[0])
        # unfixable error? problem with __reduce__ in one of the dependencies
        # return list(pool.starmap(self.query, queries))
        return [self.query(*q) for q in queries]

    #for internal testing only
    def query_cached(self, phrase, top_cut, decay_factor, elements):
        print(self.cache_mode)
        if self.cache_mode != phrase+self.pooling_mode:
            if type(phrase) == str:
                phrase_embedding, phrase_tokens = self.bc.encode([phrase], show_tokens=True)
                phrase_embedding = pooling.pool(phrase_embedding[0], self.pooling_mode,phrase_tokens[0])
            else:
                phrase_embedding = phrase[0]
                phrase_tokens = phrase[1]
            data = self.db.find_pros_with_pooling_cached(self.pooling_mode)
            print("Retrieved data")
            formatted = {}
            self.cache_mode = phrase+self.pooling_mode
            self.embed_cache = phrase_embedding
            for datum in data:
                temp = {}
                review_emb = datum[self.pooling_mode + '_pros']
                title_emb = datum[self.pooling_mode + '_title']
                rev_score = dist(phrase_embedding, review_emb)
                title_score = dist(phrase_embedding, title_emb)
                title = datum['Product title']
                temp['rev_score'] = rev_score
                temp['title_score'] = title_score
                temp['title'] = title
                temp['review'] = datum['Pros']
                try:
                    formatted[datum['href']].append(temp)
                except KeyError:
                    formatted[datum['href']] = [temp]
            for link in formatted:
                formatted[link] = sorted(formatted[link], key=lambda x: x['rev_score'])
                for i, data in enumerate(formatted[link]):
                    if i < top_cut:
                        data['rev_score'] = (1/data['rev_score']) / ((i*decay_factor)+1)
                    else:
                        break
            self.data_cache = formatted
            print("Sorted data")
        else:
            phrase_embedding = self.embed_cache
            formatted = self.data_cache
        
        links = {}
        
        for link in self.data_cache:
            score = 0
            reviews = []
            title = self.data_cache[link][0]["title"]
            for i, data in enumerate(self.data_cache[link]):
                if i < top_cut:
                    score += data['rev_score']
                    reviews.append(data['review'])
                else:
                    break
            links[link] = {"score": score, "reviews": reviews, "title": title}
        
        return sorted(links.items(), key=lambda k_v: k_v[1]['score'], reverse=True)[:elements]

    def query(self, phrase, top_cut=10, decay_factor=1, results=10):
        if type(phrase) == str:
            phrase_embedding, phrase_tokens = self.bc.encode([phrase], show_tokens=True)
            phrase_embedding = pooling.pool(phrase_embedding[0], self.pooling_mode,phrase_tokens[0])
        else:
            phrase_embedding = phrase[0]
            phrase_tokens = phrase[1]
            print(phrase_embedding, phrase_tokens)
        data = self.db.find_pros_with_pooling_cached(self.pooling_mode)
        print("Retrieved data")
        formatted = {}
        for datum in data:
            temp = {}
            review_emb = datum[self.pooling_mode + '_pros']
            title_emb = datum[self.pooling_mode + '_title']
            rev_score = dist(phrase_embedding, review_emb)
            title_score = dist(phrase_embedding, title_emb)
            title = datum['Product title']
            temp['rev_score'] = rev_score
            temp['title_score'] = title_score
            temp['title'] = title
            temp['review'] = datum['Pros']
            try:
                formatted[datum['href']].append(temp)
            except KeyError:
                formatted[datum['href']] = [temp]
        for link in formatted:
            formatted[link] = sorted(formatted[link], key=lambda x: x['rev_score'])
            for i, data in enumerate(formatted[link]):
                if i < top_cut:
                    data['rev_score'] = (1/data['rev_score']) / ((i*decay_factor)+1)
                else:
                    break

        links = {}
        
        for link in formatted:
            score = 0
            reviews = []
            title = formatted[link][0]["title"]
            for i, data in enumerate(formatted[link]):
                if i < top_cut:
                    score += data['rev_score']
                    reviews.append(data['review'])
                else:
                    break
            links[link] = {"score": score, "reviews": reviews, "title": title}
        
        return sorted(links.items(), key=lambda k_v: k_v[1]['score'], reverse=True)[:results]

    def query_hash(self, phrase, top_cut=10, decay_factor=1, results=10):
        phrase_embedding, phrase_tokens = self.bc.encode([phrase], show_tokens=True)
        phrase_embedding = pooling.pool(phrase_embedding[0], self.pooling_mode,phrase_tokens[0])
        data = self.db.find_pros_with_pooling(self.pooling_mode)
        print("Retrieved data")
        product_data = {}
        count = 0
        def get_score(record):
            nonlocal product_data
            nonlocal self
            rev_score = dist(phrase_embedding, record[self.pooling_mode + '_pros'])
            score = 1/rev_score
            key = record['href']
            try:
                if product_data[key]['score'] < score:
                    product_data[key]['score'] = score
                    product_data[key]['review'] = record['Pros']
            except KeyError:
                product_data[key] = {'title': record['Product title'], 'review': record['Pros'], 'score': score}

            nonlocal count
            count += 1
            if (count % 1000) == 0:
                print(count)
            return score

        top = heapq.nlargest(top_cut, data, key=get_score)
        return [t['href'] for t in top]

def make_file():
    pooling_modes = ["mean_pooling", "max_pooling_total",
                    "mean_pooling_pos_filtered", "max_pooling_pos_filtered_total"]
    phrases = ["warm jacket", "comfy mountain shoes",
            "large tent", "walking poles", "zapatos confortables"]

    file = open("search_title_appended_tuned_no_var.csv", "w")
    line = "mode, phrase, top_cut, decay, title, link, score\n"
    for mode in pooling_modes:
        s = Search("cotswaldsdata", mode)
        for p in phrases:
            top_cut = 10
            decay = 1
            print(mode, p, top_cut, decay)
            res = s.query_cached(p, top_cut, decay, 10)
            for link in res:
                line += mode + ", " + p + ", " + str(top_cut) + ", " + str(decay) + ", " + link[1]['title'] + ", " + link[0] + ", " + str(link[1]['score']) + ", " + '"' + link[1]['reviews'][0] + '"' + "\n"

    file.write(line)