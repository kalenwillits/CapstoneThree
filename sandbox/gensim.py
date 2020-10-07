# Testing and research on the Gensim Word2Vec model.
# imports needed and loggingimport gzipimport gensim import logging

from gensim.models import Word2Vec
from library import *
from models import *
from collections import Counter
cd_data = 'data/'
corpus = chess
w2v = Word2Vec(chess.full_tokenize, window=1)


# THis is going to get me the trained model I need.
# with open(cd_data+'train_data.txt', 'r+') as file:
#     train_doc = file.read()


with open(cd_data+'rules of chess.txt', 'r+') as file:
    train_doc = file.read()

train_article = ProcessArticle(train_doc)
# Add this to library -> ProcessArticle
corpus = dict(zip(Counter(train_article.tolest).keys(), Counter(train_article.tolest).values()))

w2v.train(corpus, total_words=len(corpus), epochs=1)

w2v.wv.similar_by_word('rule')
