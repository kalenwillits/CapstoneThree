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

def full_tokenize(doc):
    """
    Makes use of NLTK's word and sentence tokenizer on a document string and changed all
    words to lower case.
    - Returns a list of lists lower case tokenized strings.
    """
    sentences = sent_tokenize(doc)
    stop = StopWords(cd_data=cd_data)

    sentence_tokens = []
    for sentence in sentences:
        word_tokens = word_tokenize(sentence)
        word_lemons = []
        for word in word_tokens:
            if word not in stop.words:
                word_lower = word.lower()
                word_lemmatized = lemmatizer.lemmatize(word_lower)
                word_lemons.append(word_lemmatized)
        sentence_tokens.append(word_lemons)
    return sentence_tokens

    # def word_tokenize_plus(doc):
    #     word_tokens = word_tokenize(doc)
    #     word_processed = []
    #     for word in word_tokens:
    #         word_lower = word.lower()
    #         word_lemon = lemmanize(word_lower)
    #         word.
    #         word_processed.append()
    #     re



def lemmatize(doc):
    """
    Uses the previous "tokenize" function and NLTK's lemmatizer to lemmatize
    the document and return it as a list.
    """
    doc_lower = doc.lower()
    doc_word = word_tokenize(doc_lower)
    doc_lemma = []
    for word in doc_word:
        doc_lemma.append(lemmatizer.lemmatize(word))
    return doc_lemma

def tolest(doc):
    """
    Uses the previous "lemmatize" function and removes all stop words loaded in
    from the "StopWords" class. returns a list of strings with the stop words
    removed.
    """
    doc_lower = doc.lower()
    stop = StopWords(cd_data=cd_data)
    doc_stop = []
    for word in lemmatize(doc_lower):
        if word not in stop.words:
            doc_stop.append(word)
    return doc_stop

def word_tokenize_plus(doc):
    """
    Tokenizes by word, removes stop words, and lemmatizes.
    """
    stop = StopWords(cd_data=cd_data)
    word_lower = doc.lower()
    word_tokens = word_tokenize(word_lower)

    word_stop = []
    for word in word_tokens:
        if word not in stop.words:
            word_stop.append(word)

    word_lemons = []
    for word in word_stop:
        lemon = lemmatizer.lemmatize(word)
        word_lemons.append(lemon)

    return word_lemons

def sent_tokenize_plus(doc):
    doc_tolest = tolest(doc)
    doc_join = ' '.join(doc_tolest)
    doc_tokens = sent_tokenize(doc_join)
    return doc_tokens



def one_hot(doc):
    """
    Uses the previous "tolest" function and one_hot encodes them
    using the Pandas.get_dummies method. Returns a Pandas dataframe.
    """
    doc_stop = tolest(doc)
    df_1h = pd.get_dummies(doc_stop)
    return df_1h

class ProcessArticle:
    def __init__(self, doc):
        """
        Organizes the tokenize, lemmatize, tolest, and one_hot
        functions so that the document only needs to be passed when the class
        was instantiated.
        """
        self.doc = doc.lower()
        self.word_tokenize = word_tokenize(self.doc.lower())
        self.word_tokenize_plus = word_tokenize_plus(self.doc)
        self.sent_tokenize = sent_tokenize(self.doc.lower())
        self.sent_tokenize_plus = sent_tokenize_plus(self.doc)
        self.full_tokenize = full_tokenize(self.doc)
        self.lemmatize = lemmatize(self.doc)
        self.tolest = tolest(self.doc)
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


def ngrams(doc, n):
    """
    Creates n word grams.
    """
    output = []
    for i in range(len(doc)-n+1):
        gram = ' '.join(doc[i:i+n])
        if gram != '':
            output.append(gram)
    return output
