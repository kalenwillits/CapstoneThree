# sandbox/custom.py
read_article = ProcessArticle(doc)

#Gather user input
user_doc = 'The chess board has squares'

#process user input
user_article = ProcessArticle(user_doc)

grams = []
for i in range(len(user_article.tolest)):
    ngram = ngrams(user_article.tolest, i)
    grams.extend(ngram)

query_score = {}
for gram in grams:
    query_score[gram] = 0

for gram in grams:
    for sentence in article.sent_tokenize_plus:
        if gram in sentence:
            query_score[gram] += 1

# ACCURACY: the difference between the mean of the measurements and the reference value, the bias. Establishing and correcting for bias is necessary for calibration.

relevance_score = np.mean(list(query_score.values()))

gate = 20

if relevance_score < gate:
    print(False, round(relevance_score, 2))
elif relevance_score >= gate:
    print(True, round(relevance_score, 2))
