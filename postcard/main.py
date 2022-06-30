# https://colab.research.google.com/drive/1AoolDYePUpPkRCKIu0cP9zV7lX5QGD3Z?usp=sharing#scrollTo=SPbJ5Q8M_ctf

import openai
import keyword_lib as gpt
import scraper_lib



keywords,sets_keywords = scraper_lib.create_keyword(60)
gpt.gpt_init(openai)

#list(set(keywords))
#print(gpt.generate_sentence(openai,keywords))


import random
import csv
import translators
header = ['number','set_keywords','keywords','sentence','postcard','postcard_zh']
sentences = []
postcards = []
postcards_zh = []
with open('label.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    datas = []
    for i in range(100):

        input_keywords = random.choices(keywords, k=random.randint(3,10))
        input_keywords_set = list(set(input_keywords))
        sentence = gpt.generate_sentence(openai,input_keywords)
        sentences.append(sentence)

        postcard = gpt.generate_postcard(openai,input_keywords)
        postcard = postcard.replace("[Your name]","Your Friend")
        postcards.append(postcard)

        postcard_zh = translators.google(postcard,from_language='en',to_language='zh-TW')
        postcards_zh.append(postcard_zh)

        data = [
        i, #'number'
        ', '.join(input_keywords_set),#'set_keywords'
        ', '.join(input_keywords),#'keywords'
        sentence,#'sentence'
        postcard,#'postcard'
        postcard_zh#postcard-zh
        ]
        datas.append(data)

    writer.writerow(header)
    writer.writerows(datas)

#print(gpt.generate_sentence(openai,keywords))



#data = ['number','set_keywords','keywords','sentence','postcard']
