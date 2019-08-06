from bert_serving.client import BertClient
import csv
import time
bc = BertClient()
Pros_embeddings = []
Cons_embeddings = []
Title_embeddings = []
Pros = []
Cons = []
Titles = []
with open('no_duplicates_sorted.csv', encoding= "utf-8") as old:
    reader = csv.DictReader(old)
    for i, row in enumerate(reader):
        pros = row['Pros'].replace('\n', ' ||| ')
        cons = row['Cons'].replace('\n', ' ||| ')
        title = row['Product title'].replace('\n', ' ||| ')
        if pros == "":
            pros = "None"
        if cons == "":
            cons = "None"
        if title == "":
            title = "None"
        Pros.append(pros)
        Cons.append(cons)
        Titles.append(title)
    t0 = time.time()
    n = 256 * 1
    Pros = [Pros[i * n:(i + 1) * n] for i in range((len(Pros) + n - 1) // n)]
    Cons = [Cons[i * n:(i + 1) * n] for i in range((len(Cons) + n - 1) // n)]
    Titles = [Titles[i * n:(i + 1) * n] for i in range((len(Titles) + n - 1) // n)]
    for i,pros in enumerate(Pros):
        Pros_embeddings.extend(bc.encode(pros))
        print((i+1) / len(Pros))
    for i,cons in enumerate(Cons):
        Cons_embeddings.extend(bc.encode(cons))
        print((i+1) / len(Cons))
    for i,titles in enumerate(Titles):
        Title_embeddings.extend(bc.encode(titles))
        print((i+1) / len(Titles))
    
    print(Pros[0][:10])
    
    """
    Pros_embeddings = bc.encode(Pros)
    for pro in Pros[:11]:
        print(pro)
    Cons_embeddings = bc.encode(Cons)
    Title_embeddings = bc.encode(Titles)
    """

    t1 = time.time()
    print(t1-t0)


with open('no_duplicates_sorted.csv', encoding= "utf-8") as old:
    with open('simple_embeddings.csv', mode='w', encoding="utf-8") as new:
        reader = csv.DictReader(old)
        fieldnames = ['Pros', 'Cons', 'Title', 'href', 'Pros_embeddings', 'Cons_embeddings', 'Title_embeddings']
        writer = csv.DictWriter(new, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        
        Pros = [j for i in Pros for j in i]
        Cons = [j for i in Cons for j in i]
        Titles = [j for i in Titles for j in i]

        for i,row in enumerate(reader):
            new_row = {'Pros': Pros[i], 'Cons': Cons[i], 'Title': Titles[i], 'href': row['href'], 'Pros_embeddings': list(Pros_embeddings[i]), 'Cons_embeddings': list(Cons_embeddings[i]), 'Title_embeddings': list(Title_embeddings[i])}
            writer.writerow(new_row)
