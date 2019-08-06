import numpy as np
from bert_serving.client import BertClient
from pymongo import MongoClient
from collections import OrderedDict
from sortedcontainers import SortedList
from functools import reduce

bc = BertClient()
client = MongoClient()
db = client.disso
coll = db.simpleBERTcotswaldsMax


data = coll.find()

a = np.array(bc.encode(['small bottle'])[0])
best = {}
formatted = []


def dist(a, b):
    return 1-a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


for datum in data:
    temp = {}
    rev_score = dist(a, datum['Pros_embeddings'])
    title_score = dist(a, datum['Title_embeddings'])
    temp['href'] = datum['href']
    temp['rev_score'] = rev_score
    temp['title_score'] = title_score
    formatted.append(temp)

links = {datum['href'] for datum in formatted}

rev_score_by_link = OrderedDict({link: SortedList() for link in links})
title_score_by_link = OrderedDict({link: 0 for link in links})
counts = {link: 0 for link in links}


for datum in formatted:
    rev_score_by_link[datum['href']].add(datum['rev_score'])
    title_score_by_link[datum['href']] = datum['title_score']
    counts[datum['href']] += 1

top_cut = 1
ratio = 0.5
for link in rev_score_by_link:
    rev_score = (reduce(lambda x, y: x + y,
                        rev_score_by_link[link][:top_cut]) / top_cut) * ratio
    title_score = title_score_by_link[link] * (1-ratio)
    rev_score_by_link[link] = rev_score+title_score


rev_score_by_link = [{'href': link, 'score': score}
                     for (link, score) in rev_score_by_link.items()]

review_scores = [sorted(formatted, key=lambda x: x['rev_score'])[:10]]

#print(sorted(rev_score_by_link, key=lambda x: x[1])[:10])
rev_score_by_link = sorted(rev_score_by_link, key=lambda x: x['score'])
print(rev_score_by_link[:10])
print(review_scores)

#formatted = sorted(formatted, key=lambda x: x['score'])
# print(formatted[:10])

# average score from all reviews

# include product name into scoring

# show user why the decision was made
