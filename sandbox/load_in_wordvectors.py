#load_in_wordvectors.py
from library import *

SaveLoad.save(test_model.vectors, cd_models+'vectors.w2v')
from gensim.utils import SaveLoad
user_article = ProcessArticle('Unneeded input for training. Test check test amount object fox brown')
read_article = user_article
train_article = user_article
help(SaveLoad.save)

ChatBotModel(user_article=user_article, read_article=read_article, train_article=train_article)
help(KeyedVectors.load)


dummy_model = ChatBotModel(user_article=user_article,
                    read_article=chess,
                    train_article=chess,
                    train_article_name='train_sample.txt',
                    load_vectors='vectors.w2v',
                    gate=20,
                    weight_mod=1.5,
                    window=50,
                    epochs=15,
                    vector_weight=10,
                    vector_scope=5)

dummy_model.prediction
