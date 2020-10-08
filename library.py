# library.py
import wikipedia as wiki
from nltk import word_tokenize, sent_tokenize
from nltk.stem import 	WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from gensim.models import Word2Vec

# __Default Parameters__
cd_data = 'data/'
cd_figures = 'figures/'
vector_scope = 10
weight_mod = 3
vector_weight = 1
epochs = 1
window = 1
gate = 40
weight_mod = 3
grams = None
read_article = None
read_article_name = 'rules of chess'
user_article = None
train_article = None
train_article_name = None
vectors = None


def load_wiki_article(article_name=read_article_name, cd_data=cd_data):
    """
    Loads in the wiki article "rules of chess" from Wikipedia and saves it
    to a .txt file.
    """
    article = wiki.page(article_name).content
    with open(cd_data+article_name+'.txt', 'w+') as file:
        file.write(article)

def read_wiki_article(article_name=read_article_name, cd_data=cd_data):
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
    - Outputs as a list
    """
    output = []
    for i in range(len(doc)-n+1):
        gram = ' '.join(doc[i:i+n])
        if gram != '':
            output.append(gram)
    return output

def generate_grams(user_article):
    """
    Takes an article from the ProcessArticle class and generates ngrams for
    the full range of the user article.
    """
    grams = []
    for i in range(len(user_article.tolest)):
        ngram = ngrams(user_article.tolest, i)
        grams.extend(ngram)
    return grams

def calculate_query(grams=grams,
                    read_article=read_article,
                    vectors=vectors,
                    vector_scope=vector_scope,
                    weight_mod=weight_mod,
                    vector_weight=vector_weight):
    """
    Counts the number of matches in the read article when compared to the user
    article. The output is in a dictionary format with the gram as the keys and
    number of relevant counts as the value.
    """

    query_score = {}
    stop = StopWords()

    for gram in grams:
        query_score[gram] = 0

    for gram in grams:
        #
        # assert gram in stop.words, """{0} is not in the corpus! Consider
        # #                                 increasing the amount of data or adding
        # #                             {0} to stop_words.txt')""".format(gram)
        if len(gram.split(' ')) == 1:
            try:
                vectors_dict = dict(vectors.similar_by_word(gram, topn=vector_scope))
                vectors_list = list(vectors_dict)
                for key in vectors_dict.keys():
                    for sentence in read_article.sent_tokenize_plus:
                        if key in sentence:
                            query_score[gram] += vectors_dict[key]*vector_weight
            except KeyError:
                pass

        for sentence in read_article.sent_tokenize_plus:
            if gram in sentence:
                query_score[gram] += 1

    if weight_mod > 0:
        for gram in grams:
            mod_gram = len(gram.split(' '))**weight_mod
            query_score[gram] *= mod_gram


    return query_score


def calculate_relevance(query_score):
    """
    Calculates the relevance score based on the values in the query score.
    *** One known issue is if the user article is only one word, an error will
    appear here.
    """
    relevance_score = np.mean(list(query_score.values()))
    return relevance_score

def predict_doc(relevance_score, gate):
    """
    This compares the relevance score to the user defined gate and returns a
    true or false value depending on the relevancy when compared to the user
    article.
    """
    if relevance_score < gate:
        return False, relevance_score
    elif relevance_score >= gate:
        return True, relevance_score

def generate_corpus(article):
    """
    Uses the collections.Counter class to generate a corpus document on a
    ProcessArticle.tolest object.
    """
    article_keys = Counter(article).keys()
    article_values = Counter(article).values()
    article_zip = zip(article_keys, article_values)
    article_dict = dict(article_zip)

    return article_dict

def train_vectors(train_article_name,
                test_data,
                corpus,
                window=window,
                epochs=epochs,
                cd_data=cd_data):
    """
    Train doc needs to be saved to a file.

    train_article_name -> String of file name excluding the path.
    test_data -> Fully tokenized read_article
         from ProcessArticle.full_tokenize(doc).
    corpus -> Corpus dictionary
         from ProcessArticle.generate_corpus(train_article.tolest)
    window -> Default window parameter from Word2Vec.
    epochs -> Default epochs parameter from Word2vec.
    cd_data -> data directory path as a string.

    """

    with open(cd_data+train_article_name, 'r+') as file:
        train_data = file.read()

    w2v = Word2Vec(test_data, window=window)
    w2v.train(train_data, total_words=len(corpus), epochs=epochs)

    return w2v.wv




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
        self.corpus = generate_corpus(self.tolest)

class StopWords:
    def __init__(self, cd_data=cd_data):
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

class ChatBotModel:
    def __init__(self,
            user_article=user_article,
            read_article=read_article,
            train_article=train_article,
            train_article_name=train_article_name,
            cd_data=cd_data,
            gate=gate,
            weight_mod=weight_mod,
            window=window,
            epochs=epochs,
            vector_scope=vector_scope,
            vector_weight=vector_weight):
        """
        Model of the user article when compared to the read article.
        gate -> The threshold of proper relevance. A low gate will cause more
        false positives and a high gate will cause more false negatives.
        """
        self.gate = gate
        self.user_article = user_article
        self.read_article = read_article
        self.grams = generate_grams(self.user_article)

        self.vectors = train_vectors(train_article_name,
                        self.read_article.full_tokenize,
                        train_article.corpus,
                        window=window,
                        epochs=epochs,
                        cd_data=cd_data)

        self.query_score = calculate_query(grams=self.grams,
                            read_article=self.read_article,
                            vectors=self.vectors,
                            vector_scope=vector_scope,
                            weight_mod=weight_mod,
                            vector_weight=vector_weight)

        self.relevance_score = calculate_relevance(self.query_score)
        self.prediction = predict_doc(self.relevance_score, self.gate)
        self.parameters = {'gate': gate,
                            'weight_mod': weight_mod,
                            'window': window,
                            'epochs': epochs,
                            'vector_scope': vector_scope,
                            'vector_weight': vector_weight}
