import csv
import json

seen = set()
with open('simple.csv', encoding= "utf-8") as old:
    with open('simple_clean.csv', mode='w', encoding="utf-8") as new:
        reader = csv.DictReader(old)
        fieldnames = ['Pros', 'Cons', 'Product title', 'href']
        writer = csv.DictWriter(new, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            seen.add(json.dumps(row, sort_keys=True))
        for j in seen:
            writer.writerow(json.loads(j))
            
