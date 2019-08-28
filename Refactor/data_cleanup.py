import csv

def cleanup(raw_data_file_name, clean_data_file_name):
    if raw_data_file_name == "cotswolds.csv":
        with open(raw_data_file_name, encoding= "utf-8") as old:
            with open(clean_data_file_name, mode='w', encoding= "utf-8") as new:
                reader = csv.DictReader(old)
                fieldnames = ['Pros', 'Cons', 'Product title', 'href']
                writer = csv.DictWriter(new, fieldnames=fieldnames)
                writer.writeheader()
                for i,row in enumerate(reader):
                    new_row = {'Pros': row['Pros'], 'Cons': row['Cons'], 'Product title': row['Product title'], 'href': row['product selector-href']}
                    new_row['href'] = new_row['href'].split('?')[0]
                    if (new_row['Pros'] == "Reviewer left no comment"):
                        new_row['Pros'] = ""
                    if (new_row['Cons'] == "Reviewer left no comment"):
                        new_row['Cons'] = ""
                    if (new_row['Pros'] != "" or new_row['Cons'] != ""):
                        new_row['Pros'] = new_row['Pros'].replace('\n', ' ||| ')
                        new_row['Cons'] = new_row['Cons'].replace('\n', ' ||| ')
                        new_row['Pros'] = new_row['Pros'].replace('´', "'" )
                        new_row['Cons'] = new_row['Cons'].replace('´', "'")
                        writer.writerow(new_row)
    elif raw_data_file_name == "screwfix.csv":
        with open(raw_data_file_name, encoding= "utf-8") as old:
            with open(clean_data_file_name, mode='w', encoding= "utf-8") as new:
                reader = csv.DictReader(old)
                fieldnames = ['Pros', 'Cons', 'Product title', 'href']
                writer = csv.DictWriter(new, fieldnames=fieldnames)
                writer.writeheader()
                for i,row in enumerate(reader):
                    if row['review'] and row['recommended']:
                        new_row = {'Product title': row['product'], 'href': row['product-href']}
                        if row['recommended'] == "Yes":
                            new_row['Pros'] = row['review']
                            new_row['Cons'] = ""
                        elif row['recommended'] == "No":
                            new_row['Cons'] = row['review']
                            new_row['Pros'] = ""
                        try:
                            new_row['Pros']
                        except:
                            print(row)
                        if (new_row['Pros'] == "Reviewer left no comment"):
                            new_row['Pros'] = ""
                        if (new_row['Cons'] == "Reviewer left no comment"):
                            new_row['Cons'] = ""
                        if (new_row['Pros'] != "" or new_row['Cons'] != ""):
                            new_row['Pros'] = new_row['Pros'].replace('\n', ' ||| ')
                            new_row['Cons'] = new_row['Cons'].replace('\n', ' ||| ')
                            new_row['Pros'] = new_row['Pros'].replace('´', "'" )
                            new_row['Cons'] = new_row['Cons'].replace('´', "'")
                            writer.writerow(new_row)