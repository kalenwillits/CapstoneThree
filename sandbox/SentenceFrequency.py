# sandbox/SentenceFrequency.py

article = chess
data = token_counts_df


"""
Counts the frequency of words that appear in each senctence.
- designed for (chess, token_counts_df)
"""
token_sent_freq = {}
for token in data.index:
    counter = 0
    for sent in chess.sent_tokenize:
        if token.lower() in sent.lower():
            counter += 1
    token_sent_freq[token] = [counter]
df = pd.DataFrame(token_sent_freq)
df
