


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



            # #Gather user input
            # user_doc = 'The Queen can move in any direction'
            # # Process user input
            # user_article = ProcessArticle(user_doc)
            # # train_article = ProcessArticle(read_doc)
            # # Instantiate model
            #
            # model = ChatBotModel(user_article=user_article,
            #                     read_article=chess,
            #                     train_article=chess,
            #                     train_article_name='train_sample.txt',
            #                     gate=20,
            #                     weight_mod=1.5,
            #                     window=50,
            #                     epochs=15,
            #                     vector_weight=10,
            #                     vector_scope=5)

            # Save trained vectors as KeyedVectors.

            # Generate a prediction
            # user_article.tolest
            # model.prediction, model.query_score
