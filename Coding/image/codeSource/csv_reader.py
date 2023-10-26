import csv
sentences = []
with open('label.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in (reader):
        sentences.append(row['sentence'].strip())
