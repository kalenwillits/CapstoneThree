# generate_test_data.py
# Here we will read in and cannibalize the Rules of Chess wiki document and use
# that article as our baseline truth. Then we will add a random article from
# Wikipedia and use that as the false values.
from library import *
import pandas as pd
true_doc = read_wiki_article()
true_article = ProcessArticle(chess_doc)

load_wiki_article(article_name='The_Grand_Mansion_Gate')
false_doc = read_wiki_article(article_name='The_Grand_Mansion_Gate')
false_article = ProcessArticle(false_doc)

# Transform into data frames.
true_dict = {'user_doc': true_article.sent_tokenize}
true_df = pd.DataFrame(true_dict)
true_df['type'] = True

false_dict = {'user_doc': false_article.sent_tokenize}
false_df = pd.DataFrame(false_dict)
false_df['type'] = False

# Merge data frames.
df = true_df.merge(false_df, on=['user_doc', 'type'], how='outer')

# Send to file for testing.
df.to_csv(cd_data+'test_data.csv', index=False)
