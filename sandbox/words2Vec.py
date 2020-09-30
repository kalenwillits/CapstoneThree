doc = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer as ps
from nltk.stem import 	WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pandas as pd

stop = StopWords()


def process_data(doc):
    for stop_word in [',', '--', '.', '!', '\'re', '\'s', 'n\'t']:
        stop.add_word(stop_word)
    doc_sent = sent_tokenize(doc)
    # doc_word = word_tokenize(doc)
    doc_filtered = []
    for sent in doc_sent:
        sent_word = word_tokenize(sent)
        sent_filtered = []
        for word in sent_word:
            if word not in stop.words:
            # print(word)
                word_lower = word.lower()
                word_lemma = lemmatizer.lemmatize(word_lower)
                sent_filtered.append(word_lemma)
        doc_filtered.append(sent_filtered)

    return pd.DataFrame(enumerate(doc_filtered))

df = process_data(doc)

df


w	text = "studies studying cries cry"
	tokenization = nltk.word_tokenize(text)


doc_sent = sent_tokenize(doc)

doc_word = [word_tokenize(doc) for doc in doc_sent]




import matplotlib.pyplot as plt
corpus
corpus = pd.get_dummies(doc_filtered)
from gensim.models import Word2Vec
word2vec = Word2Vec(doc_filtered)

  model = Word2Vec(
        corpus,
        size=150,
        window=10,
        min_count=2,
        workers=10,
        iter=10)


# Viewing a dictionary.
vocabulary = word2vec.wv.vocab
print(vocabulary)

# Viewing a vector
v1 = word2vec.wv['e']
v1
help(word2vec)
