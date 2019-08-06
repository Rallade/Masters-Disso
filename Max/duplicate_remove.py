import csv
import json

seen = set()
with open('clean.csv', encoding= "utf-8") as old:
    with open('no_duplicates.csv', mode='w', encoding="utf-8") as new:
        reader = csv.DictReader(old)
        fieldnames = ['Pros', 'Cons', 'Product title', 'href']
        writer = csv.DictWriter(new, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            seen.add(json.dumps(row, sort_keys=True))
        for j in seen:
            writer.writerow(json.loads(j))
            
with open('no_duplicates.csv', encoding="utf-8") as old:
    with open('no_duplicates_sorted.csv', mode='w', encoding="utf-8") as new:
        all_entries = []
        reader = csv.DictReader(old)
        fieldnames = ['Pros', 'Cons', 'Product title', 'href']
        writer = csv.DictWriter(new, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(reader):
            all_entries.append(row)
        sorted_entries = sorted(all_entries, key=lambda entry: len(entry['Pros']))
        for entry in sorted_entries:
            writer.writerow(entry)