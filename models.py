# model.py

from library import *

class Model:
    def __init__(self, user_article, read_article, gate=40, weight_mod=2):
        """
        Model of the user article when compared to the read article.
        gate -> The threshold of proper relevance. A low gate will cause more
        false positives and a high gate will cause more false negatives.
        """
        self.gate = gate
        self.user_article = user_article
        self.read_article = read_article
        self.grams = generate_grams(self.user_article)
        self.query_score = calculate_query(self.grams,
            self.read_article,
            weight_mod=weight_mod)
        self.relevance_score = calculate_relevance(self.query_score)
        self.prediction = predict_doc(self.relevance_score, self.gate)
