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
# - [Model Performance Analysis](#Model-Performance-Analysis)
# - [Conclusion](#Conclusion)


# %% codecell
# __Environment__

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from gensim.models import KeyedVectors
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
rules_of_chess = ProcessArticle(doc)

# __Transform__
token_df = pd.DataFrame({'token':rules_of_chess.tolest})
token_counts_df = pd.DataFrame(token_df['token'].value_counts())
token_1h_df = rules_of_chess.one_hot

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
token_freq_df = count_token_frequency(rules_of_chess, token_counts_df)

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
number_of_tests = 100 # Number of times random parameters are generated

test_df = test_df.sample(frac=1).head(test_limit)

# Name associated reports and data files from the model's evaluation.
test_name = 'param_test_fullx10'


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
# data/validation.csv -> A dataset containing sentences from the rules of rules_of_chess
# and another random wiki article for performance measurement and model tuning.

# %% codecell
# generate random parameters for random search parameter tuning.
google_vectors = KeyedVectors.load_word2vec_format(cd_models+'GoogleNews-vectors-negative300.bin', binary=True)

metrics_columns = ['gate', 'weight_mod', 'window', 'epochs', 'vector_scope',
   'vector_weight', 'TN', 'FP', 'FN', 'TP', 'accuracy', 'precision',
   'recall', 'false_positive_rate']

metrics_dict = {}
for column in metrics_columns:
    metrics_dict[column] = []

#Generate clean test file.
pd.DataFrame(metrics_dict).to_csv(cd_data+test_name+'.csv',
                                    index=False)



for test in tqdm(range(number_of_tests)):


    parameters = {'gate': np.random.randint(3, 100),
     'weight_mod': np.random.randint(1, 5)*np.random.random(),
     'window': np.random.randint(1, 15),
     'epochs': np.random.randint(1, 25),
     'vector_scope': np.random.randint(1,25),
     'vector_weight': np.random.randint(1, 20)*np.random.random()}

    # Genereate test data and write it to csv format.

    test_model, test_df = evaluate_model(read_article=rules_of_chess,
            train_article=None, # setting as None will load google KeyedVectors
            train_article_name='train_sample.txt',
            test_df=test_df,
            load_vectors=google_vectors,
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
    metrics_df.to_csv(cd_data+test_name+'.csv',
                                            index=False,
                                            mode='a',
                                            header=False)

# Saving model params

test_group = metrics_df.groupby('accuracy').max().reset_index()
best_params = test_group[['gate', 'weight_mod', 'window', 'epochs', 'vector_scope', 'vector_weight']]
best_params.to_csv(cd_models+'parameters.csv', index=False)
parameters = best_params.transpose().to_dict()[0]

# %% codecell
# __Test Results__
print(parameters)
metrics_df.sort_values('accuracy', ascending=False)

# %% markdown
# ### Model Performance Analysis
# Ignoring that this model is likely over-fit at this point. We need to decide
# what parameters we are willing to use. To do this, let's first decide how many
# false positives we can allow. A false positive would mean the chat bot would
# tell the user a statement is true when in fact it is false. This is very bad
# and could cause extreme user frustration towards the product once put into
# production. However we must also balance false negatives for the inverse
# reason.
#
# In an attempt to avoid further over-fitting, and because we can expect the
# Word2Vec model to carry more weight when fully trained in on the Google VM,
# I will continue with the following parameters.
# ----------------------
# **After further investigation that the model was over fit due to a lack of
# specific classified data. I have since added specific inputs and classified
# them by hand to the test data. The issue with this is that it means the model
# will not be able to read in general data and will always require specific
# training to become usable. Because of this, I have opted to use Google's
# pre-trained Word2Vec model for the vectors in this algorithm. These vectors
# will provide general understanding for the model with better results, but will
# still require specfic data to learn and test on in the form of user_docs.  **

# %% codecell

# Model training done in the "model_training.py" file.

# Test user docs
user_doc = 'The game ends when a checkmate is declared.'
# user_doc = 'Pawns can move into an apartment'
# user_doc = 'The queen can move in any direction'
user_article = ProcessArticle(user_doc)

with open(cd_data+'train_sample.txt', 'r+') as file:
    train_doc = file.read()
train_article = ProcessArticle(train_doc)


trained_model = ChatBotModel(user_article=user_article,
read_article=rules_of_chess,
train_article=train_article,
train_article_name='train_data.txt',
load_vectors=google_vectors,
cd_data=cd_data,
test_df=test_df,
gate=parameters['gate'],
weight_mod=parameters['weight_mod'],
window=parameters['window'],
epochs=parameters['epochs'],
vector_scope=int(parameters['vector_scope']),
vector_weight=parameters['vector_weight'])

# %% codecell
# __Simulation__
print(user_doc, '->', trained_model.prediction)

# __Parameters After Testing on Google cloud__
# {'gate': 20.0,
#  'weight_mod': 1.74327329492608,
#  'window': 13.0,
#  'epochs': 9.0,
#  'vector_scope': 12.0,
#  'vector_weight': 14.1647849325647}


# %% markdown
#  ### Conclusion
# It's clear that a statistical model is not able to just read in an article.
# Other training steps are required. The chat bot that will be generated using
# these methods will have very little expertise on the subject until more data
# is added manually. This will require many inputs that are previously
# classified as true or false to be added to the bot's database. Since the
# model is currently acting more like a search engine, we can make use of this
# feature by making the searchable data vast and high quality. The trade off to
# this would be high amounts of training, tedious labeling, and excellent
# vectors. So for our initial purpose of implementing an all-in-one chat bot
# for support centers, legal documents, and labels, this project has shown that
# there is not an all-in-one solution. However, if the bot was trained and
# maintained to continuously become better, we could have what we are looking
# for. The bottom line is that this is not practical for the average users and
# could only be sold as a B2B service.
