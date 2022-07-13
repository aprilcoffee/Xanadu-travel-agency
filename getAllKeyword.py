import csv
sentences = []
with open('label.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in (reader):
        sentences.append(row['keywords'].strip().split(','))

sentence = []
for c in sentences:
    for i in c:
        sentence.append(i.strip().lower())


sentence = list(set(sentence))
sentence.sort()

with open('keywordList.txt', 'w') as f:
    for c in sentence:
        f.write(c)
        f.write('\n')
        #print(c,end=' ')
