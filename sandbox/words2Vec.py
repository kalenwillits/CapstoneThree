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

# 1. import NLTK, and Word2Vec -- ( re is a good idea for data cleaning)
import nltk
import re
from gensim.models import Word2Vec

# 2. Process data into tokesn.
sents = nltk.sent_tokenize(doc)
words = [nltk.word_tokenize(sent) for sent in sents]

model = Word2Vec(words, min_count=1)

model.wv.most_similar('is')
model.wv.vocab
mdoel.wv.most_simalar()

# %% codecell
# plt.plot(model.wv['purity']
# plt.plot(model.wv['Flat'])

pd.DataFrame(model.predict_output_word(['better'], topn=1))
pd.DataFrame({model.wv, model.wv.vectors})


help(model.predict_output_word)
