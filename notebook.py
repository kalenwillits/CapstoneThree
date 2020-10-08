# notebook.py
# %% markdown
# # Document Reading Chat Bot - Notebook
# Transform any document into a chat bot and ask it questions.
#
# *For the purpose of this study, we will use the "Rules of Chess" article on Wikipedia*

# %% markdown
# ## Table of Contents
# - [Data Gathering And Transforming](#Data-Gat hering-And-Transforming)
# - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
#   - [Observation](Observation)
# - [Feature Engineering](#Feature-Engineering)
# - [Testing Data](#Testing-Data)

# %% codecell
# __Environment__

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from library import *
# from models import *

cd_data = 'data/'
cd_figures = 'figures/'

# %% markdown
# ## Data Gathering And Transforming

# %% markdown
# ## Data Gathering And Transforming
# %% codecell
# __Wrangle Data__
# load_wiki_article(cd_data=cd_data) # Comment out after article has been loaded once.
doc = read_wiki_article(cd_data=cd_data)
chess = ProcessArticle(doc)

# __Transform__
token_df = pd.DataFrame({'token':chess.tolest})
token_counts_df = pd.DataFrame(token_df['token'].value_counts())
token_1h_df = chess.one_hot

# __Write To File__
token_df.to_csv(cd_data+'tokens.csv', index=False)
token_counts_df.to_csv(cd_data+'tokens_counts.csv')
token_1h_df.to_csv(cd_data+'tokens_1h.csv', index=False)

# %% markdown
# ### Exploratory Data Analysis

# %% codecell
# __Plotting The Top 10 Most Frequent Words__
data = token_counts_df.groupby('token')
plot_title = 'top-10-most-frequent-words'
plt.figure(figsize=(8,6.5))
plt.title(plot_title.replace('-', ' ').title())
plt.plot(token_counts_df['token'][:25], color='black', label='STANDARD')
plt.savefig(cd_figures+plot_title+'.png', transparent=True)
plt.xticks(rotation=45)
token_counts_df.head(25)

# %% codecell
# __Generating Word Cloud__
wordcloud = WordCloud().generate(doc)
plt.figure(figsize=(20,20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(cd_figures+'wordcloud.png', transparent=True)

# %% codecell
# __Summary Statistics__
token_counts_df.describe()

# %% codecell
# __Token Frequency Within Sentences__
token_freq_df = count_token_frequency(chess, token_counts_df)

plot_title = 'frequency-of-words-in-sentence'
plt.figure(figsize=(100,10))
plt.title(plot_title.replace('-', ' ').title())
plt.plot(token_freq_df.transpose()[0], color='black', label='STANDARD')
plt.savefig(cd_figures+plot_title+'.png', transparent=True)
plt.xticks(rotation=90)

# %% markdown
# ### Observation
# The most frequent words are not necessarily the words that are repeated.
# After checking the "Frequency Of Words In Sentence" chart, it's clear that
# words are repeated across the distribution within the same sentence.
# Rule is the said the most, but not the most repeated. Which makes sense due to
# the nature of the document.

# %% markdown
# ### Feature Engineering
#  Now that we have an idea of what the article looks like mathematically, we
# can create a model based on the input. The model I have chosen is a
# Frankenstein neural network using Word2Vec and ngrams to classify whether a
# statement is true or false.

# %% codecell
# __Prototyping__
#Gather user input
user_doc = 'The Queen can move in any direction'
# Process user input
user_article = ProcessArticle(user_doc)
# train_article = ProcessArticle(read_doc)
# Instantiate model

model = ChatBotModel(user_article=user_article,
                    read_article=chess,
                    train_article=chess,
                    train_article_name='train_sample.txt',
                    gate=30,
                    weight_mod=1.5,
                    window=50,
                    epochs=15,
                    vector_weight=10,
                    vector_scope=5)

# Generate a prediction
user_article.tolest
print(model.prediction, model.query_score)

# %% markdown
# ### Testing Data
#  Training and validation data have been gathered using the scripts
# "gather_articles.py" and "generate_validation_data.py." The data sets.
# data/train_data.txt -> A large document containing ~1000 appended wiki
# articles
# data/train_sample.txt -> A modified wiki article for fast testing.
# data/validation.csv -> A dataset containing sentences from the rules of chess
# and another random wiki article for performance measurement and model tuning.

# %% codecell
validation_df = pd.read_csv(cd_data+'validation_data.csv')
