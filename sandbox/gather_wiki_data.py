# gather_wiki_data.py
# !!!! MEANT TO BE RUN ON A VM ONLY !!!!

import numpy as np
import pandas as pd
from library import *
import wikipedia as wiki
cd_data = 'cd_data/'
with open(cd_data+'wiki_titles.txt', 'r+') as file:
    wiki_titles = file.readlines()

# Set the number of articles to gather.
num_articles = 9999

with open('train_articles.txt', 'a+') as file:
    for num in range(num_articles):
        idx = np.random.randint(len(wiki_titles))
        wiki_article = wiki.page(wiki_titles[idx]).context
        file.write(wiki_article)
