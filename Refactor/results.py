from pymongo import MongoClient
import numpy as np
from bert_serving.client import BertClient
from collections import OrderedDict
from sortedcontainers import SortedList
from functools import reduce
from pooling_modes import reduce_mean

phrase = "Breathable mountain shoes"
pooling_modes = ["mean_pooling_pros",
                 "max_pooling_single_pros", "max_pooling_total_pros"]

client = MongoClient()
db = client.disso
coll = db.cotswaldsdata
bc = BertClient()


def dist(a, b):
    if b is not None:
        return 1-a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        return 100


def simple_mean_dist(phrase):
    phrase_embedding = reduce_mean(bc.encode([phrase])[0])
    data = coll.find({'mean_pooling_pros': {"$exists": True}}, {
                     'mean_pooling_pros': 1, 'mean_pooling_title': 1, 'href': 1})
    formatted = []
    for datum in data:
        temp = {}
        review_emb = datum['mean_pooling_pros']
        title_emb = datum['mean_pooling_title']
        rev_score = dist(phrase_embedding, review_emb)
        title_score = dist(phrase_embedding, title_emb)
        temp['href'] = datum['href']
        temp['rev_score'] = rev_score
        temp['title_score'] = title_score
        formatted.append(temp)
    print("Retrieved data")
    links = {datum['href'] for datum in formatted}

    rev_score_by_link = OrderedDict({link: SortedList() for link in links})
    title_score_by_link = OrderedDict({link: 0 for link in links})
    print("Created dictionary")
    for datum in formatted:
        rev_score_by_link[datum['href']].add(datum['rev_score'])
        title_score_by_link[datum['href']] = datum['title_score']
    print("Populated dictionary")
    top_cut = 3
    ratio = 0.2
    for link in rev_score_by_link:
        rev_score = (reduce(lambda x, y: x + y,
                            rev_score_by_link[link][:top_cut]) / top_cut) * ratio
        title_score = title_score_by_link[link] * (1-ratio)
        rev_score_by_link[link] = rev_score+title_score
    print("Applied Top cut")
    rev_score_by_link = [{'href': link, 'score': score}
                         for (link, score) in rev_score_by_link.items()]
    rev_score_by_link = sorted(rev_score_by_link, key=lambda x: x['score'])
    print("Sorted")
    return rev_score_by_link[:10]


print(simple_mean_dist(phrase))
