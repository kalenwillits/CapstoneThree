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
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import SaveLoad



# __Default Parameters__
cd_data = 'data/'
cd_figures = 'figures/'
cd_models = 'models/'
vector_scope = 17.000000
weight_mod = 0.855440
vector_weight = 14.888270
epochs = 9.000000
window = 13.000000
gate = 7
grams = 1
read_article = None # ProcessArticle object
read_article_name = 'rules of chess'
user_article = None # ProcessArticle object
train_article = None # ProcessArticle object
train_article_name = 'dummy_train.txt'
train_data = 'train_sample.txt'
vectors = 1
test_df = {'user_article':[0],'type':[0],'predict':[0],'score':[0]}
test_name = 'model_metrics'
load_vectors = None
parameters = {}


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
    using the Pandas.get_dummies method. Returns a Pandas DataFrame.
    """
    doc_stop = tolest(doc)
    df_1h = pd.get_dummies(doc_stop)
    return df_1h

def count_token_frequency(article, data):
    """
    Counts the frequency of words that appear in each sentence.
    - Designed for (chess, token_counts_df)
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
    Trains the Word2Vec model using the test_data, then saves the vectors in the
    models directory.

    train_article_name -> String of file name excluding the path.
    test_data -> Fully tokenized read_article
         from ProcessArticle.full_tokenize(doc).
    corpus -> Corpus dictionary
         from ProcessArticle.generate_corpus(train_article.tolest)
    window -> Default window parameter from Word2Vec.
    epochs -> Default epochs parameter from Word2vec.
    cd_data -> data directory path as a string.

    """

    w2v = Word2Vec(test_data, window=window)
    w2v.train(train_data, total_words=len(corpus), epochs=epochs)
    SaveLoad.save(w2v, cd_models+'vectors.w2v')


    return w2v.wv

class ModelMetrics:
    def __init__(self, test_df=test_df, cd_data=cd_data):
        """
        calculates the confusion_matrix from 'type' and 'predict' columns in a test
        pandas DataFrame.
        """

        TN = len(test_df[(test_df['type'] == False)
                            &
                            (test_df['predict'] == False)])

        FP = len(test_df[(test_df['type'] == False)
                            &
                            (test_df['predict'] == True)])

        FN = len(test_df[(test_df['type'] == True)
                            &
                            (test_df['predict'] == False)])

        TP = len(test_df[(test_df['type'] == True)
                            &
                            (test_df['predict'] == True)])

        matrix_dict = {'TN': TN,
                        'FP': FP,
                        'FN': FN,
                        'TP': TP}

        self.matrix = matrix_dict
        try:
            self.accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
        except ZeroDivisionError:
            self.accuracy = 0
        try:
            self.recall = ( TP ) / ( TP + FN )
        except ZeroDivisionError:
            self.recall = 0
        try:
            self.precision = ( TP ) / ( TP + FP )
        except ZeroDivisionError:
            self.precision = 0
        try:
            self.false_positive_rate = ( FP ) / ( TN + FP )
        except ZeroDivisionError:
            self.false_positive_rate = 0

def evaluate_model(read_article=read_article,
                    train_article=train_article,
                    train_article_name=train_article_name,
                    test_df=test_df,
                    load_vectors=load_vectors,
                    test_name=test_name,
                    parameters=parameters,
                    cd_data=cd_data):
    """
    Tests the ChatBotModel using the specified parameters. Parameters and
    model performance is written to the .csv file named
    test_name+(ModelMetrics).csv. This csv can be evaluated to to find optimal
    parameters.

    read_article -> ProcessArticle object that the bot is reading from.
    train_article -> ProcessArticle object that the bot is taught from.
    train_article_name -> Title of training article
    test_df -> Pandas DataFrame with user_doc and type features.
    test_name -> Name of the generated test file containing the output data.
    parameters -> Dictionary matching the ChatBotModel parameters.
    cd_data -> Path to output directory.



    Returns the test DataFrame.
    """

    for user_doc in test_df['user_doc']:
        user_article = ProcessArticle(user_doc)
        test_model = ChatBotModel(user_article=user_article,
                            read_article=read_article,
                            train_article=train_article,
                            train_article_name=train_article_name,
                            gate=parameters['gate'],
                            load_vectors=load_vectors,
                            weight_mod=parameters['weight_mod'],
                            window=parameters['window'],
                            epochs=parameters['epochs'],
                            vector_weight=parameters['vector_weight'],
                            vector_scope=parameters['vector_scope'])
        idx = test_df[test_df['user_doc'] == user_doc].index[0]
        test_df.loc[idx, 'predict'] = test_model.prediction[0]
        test_df.loc[idx, 'score'] = test_model.prediction[1]

    return test_model, test_df

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
            load_vectors=load_vectors,
            cd_data=cd_data,
            test_df=test_df,
            gate=gate,
            weight_mod=weight_mod,
            window=window,
            epochs=epochs,
            vector_scope=vector_scope,
            vector_weight=vector_weight):
        """
        Model of the user article when compared to the read article.
        Utilized an ngrams loop and Gensim's Word2Vec to compare word meaning.

        user_article -> ProcessArticle object from the user_doc.

        read_article -> ProcessArticle object from which the model is reading
        from.

        train_article -> ProcessArticle object from which the model is trained
        on.

        train_article_name -> Title of the article that the model will train on.
        This variable is used for downloading the artilce from Wikipedia and
        saving the article to a file.

        cd_data -> Path to the data directory as a string.

        gate -> The threshold of proper relevance. A low gate will cause more
        false positives and a high gate will cause more false negatives.

        weight_mod -> A float number that is used as an exponent in the ngrams
        loop where n**weight_mod = relevance_score.

        window -> Parameter passed down to the Word2Vec model selecting the
        range of neighbors calculated around the selected number.

        epochs -> Parameter passed down to the Word2Vec.train() method. This
        determines the number of times the documents will be looped over in the
        training process.

        vector_scope -> Chooses the number of words to count in the query_score
        produced from the Word2Vec.wv.similar_by_word method. This parameter is
        aliased from topn.

        vector_weight -> The float number that the counter of vectors is
        multiplied by to determine what the vector weight is worth in the
        classification.

        """
        self.gate = gate
        self.weight_mod = weight_mod
        self.window = window
        self.epochs = epochs
        self.vector_scope = vector_scope
        self.vector_weight = vector_weight
        self.user_article = user_article
        self.read_article = read_article
        self.grams = generate_grams(self.user_article)
        if load_vectors == None:
            self.vectors = train_vectors(train_article_name,
                            self.read_article.full_tokenize,
                            train_article.corpus,
                            window=window,
                            epochs=epochs,
                            cd_data=cd_data)
        else:
            self.vectors = load_vectors.wv

        self.query_score = calculate_query(grams=self.grams,
                            read_article=self.read_article,
                            vectors=self.vectors,
                            vector_scope=vector_scope,
                            weight_mod=weight_mod,
                            vector_weight=vector_weight)

        self.relevance_score = calculate_relevance(self.query_score)
        self.prediction = predict_doc(self.relevance_score, self.gate)
        self.parameters = {'gate': self.gate,
                            'weight_mod': self.weight_mod,
                            'window': self.window,
                            'epochs': self.epochs,
                            'vector_scope': self.vector_scope,
                            'vector_weight': self.vector_weight}
