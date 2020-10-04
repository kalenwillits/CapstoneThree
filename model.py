# model.py

import numpy as np

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

def generate_grams(user_article):
    """
    """
    grams = []
    for i in range(len(user_article.tolest)):
        ngram = ngrams(user_article.tolest, i)
        grams.extend(ngram)
    return grams

def calculate_query(grams, read_article):
    """
    """
    query_score = {}
    for gram in grams:
        query_score[gram] = 0

    for gram in grams:
        for sentence in read_article.sent_tokenize_plus:
            if gram in sentence:
                query_score[gram] += 1

    return query_score


def calculate_relevance(query_score):
    """
    """
    relevance_score = np.mean(list(query_score.values()))
    return relevance_score

def predict_doc(relevance_score, gate):
    """
    """
    if relevance_score < gate:
        return False, relevance_score
    elif relevance_score >= gate:
        return True, relevance_score


def listen(prompt=' -> '):
    """
    """
    user_doc = input(prompt)
    return user_doc

def respond(response, relevance_score, prompt=' <- ', metrics=False):
    """
    """
    if metrics == True:
        return prompt+str(response)+'\n'+str(relevance_score)
    elif metrics == False:
        return prompt+str(response)

class Model:
    def __init__(self, user_article, read_article, gate=20):
        """
        """
        self.gate = gate
        self.user_article = user_article
        self.read_article = read_article
        self.grams = generate_grams(self.user_article)
        self.query_score = calculate_query(self.grams, self.read_article)
        self.relevance_score = calculate_relevance(self.query_score)
        self.prediction = predict_doc(self.relevance_score, self.gate)
        self.response = respond(self.prediction, self.relevance_score)
