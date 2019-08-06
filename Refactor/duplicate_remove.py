import csv
import json


def remove_dupes(clean_csv_file_name, new_file_name):
    seen = set()
    with open(clean_csv_file_name, encoding= "utf-8") as old:
        with open(new_file_name, mode='w', encoding="utf-8") as new:
            reader = csv.DictReader(old)
            fieldnames = ['Pros', 'Cons', 'Product title', 'href']
            writer = csv.DictWriter(new, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(reader):
                seen.add(json.dumps(row, sort_keys=True))
            for j in seen:
                writer.writerow(json.loads(j))
