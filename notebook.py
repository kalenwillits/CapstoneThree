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
# = [Model Performance Analysis](#Model-Performance-Analysis)

# %% codecell
# __Environment__

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import os
from library import *

cd_data = 'data/'
cd_figures = 'figures/'
cd_docs = 'docs/'
cd_models = 'models/'
# %% markdown
# ### Data Gathering And Transforming
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
# __Model Creation & Testing__
# Import test data
test_df = pd.read_csv(cd_data+'test_data.csv')
test_df['predict'] = pd.NA
test_df['score'] = pd.NA

# Test parameters. Limits the number of test runs.
test_limit = len(test_df) # for full use of the dataset use len(test_df)
number_of_tests = 50 # Number of times random paramters are generated

test_df = test_df.sample(frac=1).head(test_limit)

# Name associated reports and data files from the model's evaluation.
test_name = 'param_test_x50'



# %% markdown
# ### Testing Data
#  Training and validation data have been gathered using the scripts
# "gather_articles.py" and "generate_validation_data.py." The data sets.
#
# data/train_data.txt -> A large document containing ~1000 appended wiki
# articles
#
# data/train_sample.txt -> A modified wiki article for fast testing.
#
# data/validation.csv -> A dataset containing sentences from the rules of chess
# and another random wiki article for performance measurement and model tuning.

# %% codecell
# generate random parameters for random search parameter tuning.

with open(cd_data+'train_sample.txt') as file:
    train_sample_doc = file.read()

train_sample_article = ProcessArticle(train_sample_doc)

metrics_columns = ['gate', 'weight_mod', 'window', 'epochs', 'vector_scope',
   'vector_weight', 'TN', 'FP', 'FN', 'TP', 'accuracy', 'precision',
   'recall', 'false_positive_rate']

metrics_dict = {}
for column in metrics_columns:
    metrics_dict[column] = []

#Generate clean test file.
pd.DataFrame(metrics_dict).to_csv(cd_data+test_name+'(ModelMetrics).csv',
                                    index=False)



for test in tqdm(range(number_of_tests)):


    parameters = {'gate': np.random.randint(3, 100),
     'weight_mod': np.random.randint(1, 5)*np.random.random(),
     'window': np.random.randint(1, 15),
     'epochs': np.random.randint(1, 25),
     'vector_scope': np.random.randint(1,25),
     'vector_weight': np.random.randint(1, 20)*np.random.random()}

    # Genereate test data and write it to csv format.

    test_model, test_df = evaluate_model(read_article=chess,
            train_article=train_sample_article,
            train_article_name='train_sample.txt',
            test_df=test_df,
            parameters=parameters,
            test_name=test_name)

    metrics = ModelMetrics(test_df=test_df)
    metrics_dict['gate'].append(test_model.gate)
    metrics_dict['weight_mod'].append(test_model.weight_mod)
    metrics_dict['window'].append(test_model.window)
    metrics_dict['epochs'].append(test_model.epochs)
    metrics_dict['vector_scope'].append(test_model.vector_scope)
    metrics_dict['vector_weight'].append(test_model.vector_weight)
    metrics_dict['TN'].append(metrics.matrix['TN'])
    metrics_dict['FP'].append(metrics.matrix['FP'])
    metrics_dict['FN'].append(metrics.matrix['FN'])
    metrics_dict['TP'].append(metrics.matrix['TP'])
    metrics_dict['accuracy'].append(metrics.accuracy)
    metrics_dict['precision'].append(metrics.precision)
    metrics_dict['recall'].append(metrics.recall)
    metrics_dict['false_positive_rate'].append(metrics.false_positive_rate)

        # test_df.to_csv(cd_data+test_name+'.csv', index=False)



        # initializing DataFrame.

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(cd_data+test_name+'(ModelMetrics).csv',
                                            index=False,
                                            mode='a',
                                            header=False)

# Saving model vectors
# metrics_df = pd.read_csv(cd_data+test_name+'(ModelMetrics).csv')
test_df
help(metrics_df.groupby)
test_group = metrics_df.groupby('accuracy').max().reset_index()
test_group.sort_values('accuracy', ascending=False).iloc[2]
# %% markdown
# ### Model Performance Analysis
# Ignoring that this model is likely over-fit at this point. We need to decide
# what parameters we are willing to use. To do this, let's first decide how many
# false positives we can allow. A false positve would mean the chat bot would
# tell the user a statment is true when in fact it is false. This is very bad
# and could cause extreme user frustration towards the product once put into
# production. However we must also balance false negatives for the inverse
# reason.
#
# In an attempt to avoid further over-fitting, and because we can expect the
# Word2Vec model to carry more weight when fully trained in on the Google VM,
# I will continue with the following parameters.

# >accuracy                 0.955056
# >gate                     7.000000
# >weight_mod               0.855440
# >window                  13.000000
# >epochs                   9.000000
# >vector_scope            17.000000
# >vector_weight           14.888270
# >TN                      75.000000
# >FP                       1.000000
# >FN                      15.000000
# >TP                     265.000000
# >precision                0.996241
# >recall                   0.946429
# >false_positive_rate      0.013158

# %% codecell
# Train and save
# model.vectors.save(os.path.join('models', 'vectors.w2v'))
