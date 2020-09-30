# Notes

Pseudo-Code

bert_ NOGO

GOOGLE ASSISTANT API
Word2Vec
- Wrangle your data
- Learn Google's cloud platform


Use google's word2vec pre-trained model.

_____________________________________________________
1. Gather user input
2. Transform user input -- ( Tokenize, Stemming, Vectors, remove stop words)
4. Read in Document data
5. Transform Document Data
6. Train classification model on document data. -- (BERT?)
7. Predict on user input
8. return prediction (classification as True or False)
_____________________________________________________

EDA will be soft-coded with a sample question. Actual model will be created with some-sort of UI.

If possible, lets let the user ask a question rather than provide a statement.

Sample question for jupyter-notebook.
- Use 'Rules of Chess' from Wikipedia API.

Tiers of completion:
1. Chat bot will read in document and user input as a statement, then return True or False.
2. Chat bot will read in document and user input as a question, then return yes or no.
3. Chat bot will be able to read in document and user input as a question, then return answer as full sentence.

Need to use Parts-Of-Speech Tagger ( POS Tagger ) to classify intent.

Create several types of DataFrames.
- Word Vectors ( powered by Spacy's en_core_web_sm)
- word tokens

TODO:
- Remove stop-words and punctuation from data.
- Gather and create several dataframes
- Create flow chart and sudo-code project plan.
