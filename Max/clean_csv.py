import csv

with open('Full cotswolds v1.csv', encoding= "utf-8") as old:
    with open('clean.csv', mode='w', encoding= "utf-8") as new:
        reader = csv.DictReader(old)
        fieldnames = ['Pros', 'Cons', 'Product title', 'href']
        writer = csv.DictWriter(new, fieldnames=fieldnames)
        writer.writeheader()
        for i,row in enumerate(reader):
            new_row = {'Pros': row['Pros'], 'Cons': row['Cons'], 'Product title': row['Prduct title'], 'href': row['product selector-href']}
            new_row['href'] = new_row['href'].split('?')[0]
            if (new_row['Pros'] == "Reviewer left no comment"):
                new_row['Pros'] = ""
            if (new_row['Cons'] == "Reviewer left no comment"):
                new_row['Cons'] = ""
            if (new_row['Pros'] != "" or new_row['Cons'] != ""):
                writer.writerow(new_row)