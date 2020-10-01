# notebook.py
# %% markdown
# # Document Reading Chat Bot - Notebook
# - Transform any document into a chat bot and ask it questions.

# %% markdown
# ## Table of Contents
# - [Data Gathering And Transforming](#Data-Gathering-And-Transforming)
# - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
#   - [Observation](Observation)

# %% codecell
# __Environment__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from library import *

cd_data = 'data/'
cd_figures = 'figures/'

# %% markdown
# ## Data Gathering And Transforming

# %% markdown
# ## Data Gathering And Transforming
# %% codecell
# __Wrangle Data__
# load_wiki_article(cd_data=cd_data)
doc = read_wiki_article(cd_data=cd_data)
chess = ProcessedArticle(doc)

# __Transform__
token_df = pd.DataFrame({'token':chess.remove_stopwords})
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
