# %% markdown
# Document Reading Chat Bot
# - Transform any document into a chat bot and ask it questions.

# %% codecell
# __Environment__
import pandas as pd
from library import *
from tqdm import tqdm

cd_data = 'data/'
cd_figures = 'figures/'

# %% codecell
# __Wrangle Data__
load_wiki_article(cd_data=cd_data)
doc = read_wiki_article(cd_data=cd_data)
chess = ProcessedArticle(doc)

# __Transform__
token_df = pd.DataFrame({'token':chess.tokenize})
token_counts_df = pd.DataFrame(token_df['token'].value_counts())
token_1h_df = chess.one_hot

# __Write To File__
token_df.to_csv(cd_data+'tokens.csv', index=False)
token_counts_df.to_csv(cd_data+'tokens_counts.csv', index=False)
token_1h_df.to_csv(cd_data+'tokens_1h.csv', index=False)
