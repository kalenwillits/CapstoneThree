
cd_data='data/'; article_name='Rules of chess'; add_stop_words=[]


stop = StopWords()
for stop_word in add_stop_words:
    stop.add_word(stop_word)

with open(cd_data+article_name+'.txt', 'r+') as file:
    doc = file.read()


doc_word = word_tokenize(doc)

doc_lemma = []
for word in doc_word:
    doc_lemma.append(lemmatizer.lemmatize(word))

doc_stop = []
for word in doc_lemma:
    if word not in stop.words:
        doc_stop.append(word)

doc_1h = pd.get_dummies(doc_stop)


doc_1h.sum().head(50)
