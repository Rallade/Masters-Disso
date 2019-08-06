from bert_serving.client import BertClient
import numpy as np
bc = BertClient()
print((bc.encode(["hello", "bonjour, yo"], show_tokens=True)[0][0]))

"""from sklearn.metrics.pairwise import pairwise_distances

a = np.array(bc.encode(['No'])[0])
b = np.array(bc.encode([''])[0])

print(a.shape, b.shape)
def dist1(a,b):
     return np.linalg.norm(a-b)
    
def dist2(a,b):
     return 1-a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(dist1(a,b), dist2(a,b))"""