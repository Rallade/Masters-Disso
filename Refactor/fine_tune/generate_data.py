import random

def generate(data, fraction):
    titles = set()
    print("starting")
    with open('training_data_titles_reviews.txt', mode='w', encoding="utf-8") as training_data:
        for datum in data:
            if random.random() < fraction:
                titles.add(datum["Product title"])
                try:
                    titles.add(datum["Pros"])
                except:
                    pass
                try:
                    titles.add(datum["Cons"])
                except:
                    pass
        for title in titles:
            training_data.write(title+"\n\n")
