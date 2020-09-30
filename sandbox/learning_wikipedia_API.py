import wikipedia as wiki
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer as ps
import pandas as pd

help(ps.stem)

ps = PorterStemmer()
ps.stem('driving')
print(wiki.search('Tomatoes are red'))


chess = wiki.page('Rules of Chess').content
chess_word_tokens = word_tokenize(chess)
chess_sent_tokens = sent_tokenize(chess)

chess_df = pd.DataFrame(chess_word_tokens)
wiki.summary('Linux')
wiki.page('Python Programming').title()
# Grabs complete content as a string

battles = word_tokenize(wiki.page('Battles').content)

nltk.download()
sent_tokenize(battles)
stopwords

battles = list(filter(lambda w: w != '', battles))
battles
wordsFiltered = []
for w in battles:
    if w not in stopwords.words():
        wordsFiltered.append(ps.stem(w))

import pandas as pd
df = pd.DataFrame(wordsFiltered, columns=['word'])
df['word'].value_counts()

stopwords.words('english')
battles

help(wiki.search)
