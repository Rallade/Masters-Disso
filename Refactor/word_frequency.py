from pymongo import MongoClient
client = MongoClient()
db = client.disso
coll = db.cotswaldsdata
from add_pooling_to_DB import remake_tokens, simplify_nested_embeddings
from nltk.corpus import stopwords
cursor = coll.find()

sw = stopwords.words('english')

count = {}

for record in cursor:
    try:
        pros_tokens, pros_embeddings = remake_tokens(record['pros_tokens'], record['full_pros_embedding'])
        for token in pros_tokens:
            try:
                count[token] += 1
            except KeyError:
                if token.isalnum():
                    count[token] = 1
    except:
        pass
    try:
        cons_tokens, cons_embeddings = remake_tokens(record['cons_tokens'], record['full_cons_embedding'])
        for token in cons_tokens:
            try:
                count[token] += 1
            except KeyError:
                if token.isalnum():
                    count[token] = 1
    except:
        pass


sorted_counts = sorted(count.items(), key=lambda x: x[1], reverse=True)
interest = sorted_counts[:30]
print(interest)
y = range(len(interest))
import matplotlib.pyplot as plt

plt.bar(y, [i[1] for i in interest])
plt.xticks(y, [i[0] for i in interest])
plt.xlabel('Word', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.show()
#plt.bar(y, [])