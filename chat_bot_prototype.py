# Prototype Chat Bot
from time import sleep
from library import *
from models import *
from subprocess import call
pause_time = 0.5

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
print('-> Okay I\'ve got it, what would you like to know?')
sleep(pause_time)

# Start loop
loop = True
while loop:
    user_doc = input('\n<- ').lower()
    if user_doc == 'bye':
        sleep(pause_time)
        print('-> Goodbye!')
        sleep(pause_time)
        call('clear')
        loop = False
    else:
        user_article = ProcessArticle(user_doc)
        model = Model(user_article, read_article)
        sleep(pause_time)
        print('\n-> That is', str(model.prediction[0]).lower()+'.')
