# %% codecell
# library.py
import wikipedia as wiki
from nltk import word_tokenize, sent_tokenize
from nltk.stem import 	WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

cd_data = 'data/'
cd_figures = 'figures/'

lemmatizer = WordNetLemmatizer()


class StopWords:
    def __init__(self, cd_data=''):
        """
        Loads in stops words as a list from the "stop_words.txt file" as a list.
        """
        with open(cd_data+'stop_words.txt', 'r+') as file:
            self.words = [word.strip('\n').lower() for word in file.readlines()]

    def add_word(self, words):
        """
        Adds a word to the in-memory list.
        This does not write to the stop_words.txt file
         """
        self.words.append(words)

    def summary(self):
        """
        Prints summary information about stop the stop words list.
        - Currently only reporting number of words.
        """
        print('Number of words: ', len(self.words))

def load_wiki_article(article_name='Rules of chess', cd_data=''):
    """
    Loads in the wiki article "Rules of chess" from Wikipedia and saves it
    to a .txt file.
    """
    article = wiki.page(article_name).content
    with open(cd_data+article_name+'.txt', 'w+') as file:
        file.write(article)

def read_wiki_article(article_name='Rules of chess', cd_data=''):
    """
    Reads the wiki article off of the saved .txt file and
    returns it as a string.
    """
    with open(cd_data+article_name+'.txt', 'r+') as file:
        article = file.read()
    return article

def tokenize(doc):
    """
    Makes use of NLTK's words tokenizer on a document string and changed all
    words to lower case.
    - Returns a list of lower case tokenized strings.
    """
    doc_word = word_tokenize(doc.lower())
    return doc_word

def lemmatize(doc):
    """
    Uses the previous "tokenize" function and NLTK's lemmatizer to lemmatize
    the document and return it as a list.
    """
    doc_word = word_tokenize(doc)
    doc_lemma = []
    for word in doc_word:
        doc_lemma.append(lemmatizer.lemmatize(word))
    return doc_lemma

def remove_stopwords(doc):
    """
    Uses the previous "lemmatize" function and removes all stop words loaded in
    from the "StopWords" class. returns a list of strings with the stop words
    removed.
    """
    stop = StopWords(cd_data=cd_data)
    doc_stop = []
    for word in lemmatize(doc):
        if word.lower() not in stop.words:
            doc_stop.append(word)
    return doc_stop

def one_hot(doc):
    """
    Uses the previous "remove_stopwords" function and one_hot encodes them
    using the Pandas.get_dummies method. Returns a Pandas dataframe.
    """
    doc_stop = remove_stopwords(doc)
    df_1h = pd.get_dummies(doc_stop)
    return df_1h

class ProcessedArticle:
    def __init__(self, doc):
        """
        Organizes the tokenize, lemmatize, remove_stopwords, and one_hot
        functions so that the document only needs to be passed when the class
        was instantiated.
        """
        self.doc = doc
        self.tokenize = tokenize(self.doc)
        self.sent_tokenize = sent_tokenize(self.doc)
        self.lemmatize = lemmatize(self.doc)
        self.remove_stopwords = remove_stopwords(self.doc)
        self.one_hot = one_hot(self.doc)


def batch_data(data, num_batches=10):
    """
    Splits an array into batches of a specified size.
    """
    batch_size = round(len(data)/num_batches)
    batches = []
    size = len(data)
    steps = np.arange(0, size, batch_size).tolist()
    idx = 0

    while idx < len(steps):
        if steps[idx] == steps[-1]:
            break
        batch_df = data[steps[idx]:steps[idx+1]]
        batches.append(batch_df)
        idx += 1

    print('Batch Size: ', batch_size,
    '\nBatches: ', num_batches,
    '\nOriginal: ', size)
    return batches

def count_token_frequency(article, data):
    """
    Counts the frequency of words that appear in each senctence.
    - designed for (chess, token_counts_df)
    - Returns as a Pandas DataFrame.
    """
    token_sent_freq = {}
    for token in data.index:
        counter = 0
        for sent in article.sent_tokenize:
            if token.lower() in sent.lower():
                counter += 1
        token_sent_freq[token] = [counter]
    df = pd.DataFrame(token_sent_freq)
    return df
