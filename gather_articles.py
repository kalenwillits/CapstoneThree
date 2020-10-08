# gather_articles.py
# Script to gather the top 1000 articles from wikipedia as of October 7th, 2020.
# Run time was ~45 Minutes on a Linux machine using an i7 and 16GB of RAM.

import wikipedia as wiki
import pandas as pd
import re
from tqdm import tqdm

cd_data = 'data/'

df = pd.read_csv(cd_data+'top_1000_wiki.csv', encoding = "ISO-8859-1", names=['Title'])

# Cleaning metrics off of article titles.
wikipedia_titles = []
for title in df['Title']:
    title_endnums = re.sub(r'\s\d+\s\d+', '', title)
    title_startnums = re.sub(r'\d+\s', '', title_endnums)
    wikipedia_titles.append(title_startnums)

#Placing all articles into one string.
wiki_articles = ''
for title in tqdm(wikipedia_titles):
    # Try statement avoids any article names that are no longer formatted correctly.
    try:
        article = wiki.page(title).content
        wiki_articles += article
    except:
        continue

# Writing the articles to a file for model training.
with open(cd_data+'train_data.txt', 'w+') as file:
    file.write(wiki_articles)
