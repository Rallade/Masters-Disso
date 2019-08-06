import numpy as np
from bert_serving.client import BertClient
from pymongo import MongoClient
from collections import OrderedDict
from sortedcontainers import SortedList
from functools import reduce

bc = BertClient()
client = MongoClient()
db = client.disso
coll = db.simpleBERTcotswalds


data = coll.find()

a = np.array(bc.encode(['warm jacket'])[0])
best = {}
best_score = 1
formatted = []

def dist(a,b):
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

avg_scores = OrderedDict({link: SortedList() for link in links})
counts = {link: 0 for link in links}

for datum in formatted:
     avg_scores[datum['href']].add(datum['rev_score'])
     counts[datum['href']] += 1

top_cut = 2

for link in avg_scores:
     avg_scores[link] = reduce(lambda x, y: x + y, avg_scores[link][:top_cut]) / top_cut


avg_scores = [{'href': link, 'score': score} for (link,score) in avg_scores.items()]

review_scores = [sorted(formatted, key=lambda x: x['rev_score'])[:10]]

#print(sorted(avg_scores, key=lambda x: x[1])[:10])
avg_scores = sorted(avg_scores, key=lambda x: x['score'])
print(avg_scores[:10])
#print(review_scores)

#formatted = sorted(formatted, key=lambda x: x['score'])
#print(formatted[:10])

#average score from all reviews

#include product name into scoring

#show user why the decision was made