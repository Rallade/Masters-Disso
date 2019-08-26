import numpy as np
from bert_serving.client import BertClient
from collections import OrderedDict
from sortedcontainers import SortedList
from functools import reduce
import pooling
import db_helpers

pooling_modes = ["mean_pooling", "max_pooling_single", "max_pooling_total",
                 "mean_pooling_pos_filtered", "max_pooling_pos_filtered_single", "max_pooling_pos_filtered_total"]

bc = BertClient()
db = db_helpers.DB_helpers("cotswaldsdata_not_tuned")

embed_cache = []
data_cache = []
cache_mode = ""


def dist(a, b):
    if b is not None:
        return 1-a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        return 1


def simple_dist(phrase, pooling_mode, top_cut, ratio):
    # "mean_pooling" "max_pooling_single" "max_pooling_total"
    global cache_mode
    global embed_cache
    global data_cache
    print(cache_mode)
    if cache_mode != phrase+pooling_mode:
        phrase_embedding, phrase_tokens = bc.encode([phrase], show_tokens=True)
        phrase_embedding = pooling.pool(phrase_embedding[0], pooling_mode,phrase_tokens[0])
        data = db.find_pros_with_pooling_cached(pooling_mode)
        formatted = []
        cache_mode = phrase+pooling_mode
        embed_cache = phrase_embedding
        for datum in data:
            temp = {}
            review_emb = datum[pooling_mode + '_pros']
            title_emb = datum[pooling_mode + '_title']
            rev_score = dist(phrase_embedding, review_emb)
            title_score = dist(phrase_embedding, title_emb)
            title = datum['Product title']
            temp['href'] = datum['href']
            temp['rev_score'] = rev_score
            temp['title_score'] = title_score
            temp['title'] = title
            formatted.append(temp)
            data_cache = formatted
        print("Retrieved data")
    else:
        phrase_embedding = embed_cache
        formatted = data_cache
    print("Parsed data")
    links = {datum['href'] for datum in formatted}

    rev_score_by_link = OrderedDict({link: SortedList() for link in links})
    title_score_by_link = OrderedDict({link: 0 for link in links})
    title_by_link = {}
    #print("Created dictionary")
    for datum in formatted:
        rev_score_by_link[datum['href']].add(datum['rev_score'])
        title_score_by_link[datum['href']] = datum['title_score']
        title_by_link[datum['href']] = datum['title']
    #print("Populated dictionary")
    #top_cut = 3
    #ratio = 0.2
    for link in rev_score_by_link:
        rev_score = (reduce(lambda x, y: x + y,
                            rev_score_by_link[link][:top_cut]) / top_cut) * ratio
        title_score = title_score_by_link[link] * (1-ratio)
        rev_score_by_link[link] = rev_score+title_score
    #print("Applied Top cut")
    rev_score_by_link = [{'href': link, 'score': score}
                         for (link, score) in rev_score_by_link.items()]
    for score in rev_score_by_link:
        score['title'] = title_by_link[score['href']]
    rev_score_by_link = sorted(rev_score_by_link, key=lambda x: x['score'])
    # print(rev_score_by_link[0])
    # print("Sorted")
    return rev_score_by_link[:10]


phrases = ["warm jacket", "comfy mountain shoes",
           "large tent", "walking poles", "zapatos confortables"]

file = open("variable_results_db_not_tuned.csv", "w")

line = "mode, top_cut, ratio, title, link, score\n"
for mode in pooling_modes:
    for p in phrases:
        for top_cut in range(1, 10):
            for ratio in np.arange(0.0, 1.1, 0.25):
                print(mode, p, top_cut, ratio)
                res = simple_dist(p, mode, top_cut, ratio)
                for link in res:
                    line += mode + ", " + p + ", " + str(top_cut) + ", " + str(ratio)[:4] + ", " + link['title'] + ", " + link['href'] + ", " + str(link['score']) + "\n"

file.write(line)
