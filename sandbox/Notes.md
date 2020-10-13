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
<!-- - Remove stop-words and punctuation from data. -->
<!-- - Gather and create several dataframes -->
<!-- - Create flow chart and sudo-code project plan. -->
<!-- - Create prototype of chatbot in terminal -->
<!-- - Add larger weights to larger grams in the calculate_query function -->
<!-- - ***Find a way for more relevant words to be worth more weight in the score. ( Maybe a gradient boosting technique ( canceled, the weights for larger ngrams should be sufficient) -->
<!-- - Add create corpus function to process article in library. -->
<!-- Use the word2vec vectors in the calculation to find synonyms.
- Use the code in Gensim.py to do this. -->
<!-- Create a validation data set for metrics. -->
<!-- - Use this data to create a confusion matrix for measuring model performance. -->
<!-- - generate model output for later recall -->
<!-- Update Doc strings -->
<!-- Generate report on model metrics and parameters. -->
<!-- Use random search to search for optimal model performance  -->
<!-- Save trained Word2vec model for later import -->
<!-- Initialize csv for model metrics. -->
<!-- Add data to the test set.  -->
<!-- Change the load_vectors param to pass a Word2Vec Model. ( So it does not have to train or load every time.) -->
<!-- Update prototype ( for presentation ) -->
Write readme
- Jargon
- include tree
Write model params doc
Create Excel Tables for presentation
Use Metabase to analyze performance data.
