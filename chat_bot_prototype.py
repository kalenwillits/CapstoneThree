# Prototype Chat Bot
from time import sleep
from library import *
from models import *
from subprocess import call
import pandas as pd
from gensim.models import KeyedVectors
pause_time = 0.5
cd_models = 'models/'
parameters_df = pd.read_csv(cd_models+'parameters.csv')
parameters = parameters_df.transpose().to_dict()[0]

with open(cd_data+'train_data.txt') as file:
    train_data_doc = file.read()

train_article = ProcessArticle(train_data_doc)

google_vectors = KeyedVectors.load_word2vec_format(cd_models+'GoogleNews-vectors-negative300.bin', binary=True)

call('clear')
print('-> Hello!')
sleep(pause_time)
print('-> I am the prototype chat bot interface for Wikipedia documents.')
sleep(pause_time)
print('-> What topic would you like to discuss?')
sleep(pause_time)
topic = input('\n<- ').lower()
sleep(pause_time)
print('\n-> Please allow me a moment to train on', topic+'.')
load_wiki_article(article_name=topic, cd_data=cd_data)
read_doc = read_wiki_article(article_name=topic, cd_data=cd_data)
read_article = ProcessArticle(read_doc)
print('-> Okay I\'ve got it, tell me fact and I\'ll tell you if the article I read supports your fact.')
sleep(pause_time)

# Start loop
loop = True
while loop:
    user_doc = input('\n<- ').lower()
    if user_doc == 'bye':
        sleep(pause_time)
        print('\n-> Goodbye!')
        sleep(pause_time)
        call('clear')
        loop = False
    else:
        user_article = ProcessArticle(user_doc)
        model = ChatBotModel(user_article=user_article,
        read_article=read_article,
        train_article=None,
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
        sleep(pause_time)
        print('\n-> That is', str(model.prediction[0]).lower()+'.')
