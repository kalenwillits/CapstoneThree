<!-- readme.md -:
# ChatBot Document Reader
## Changing Any Document Into An Interactive Chat Bot.

### Project Summary
The purpose was to automate user interaction and increase understanding by
proving any document could be read by a chat bot and become the new interface
for that document. My hope, was design a method for creating  a tool that limits
customer support interactions and provide an automated service to customers at a
fraction of the cost. If this worked as designed, personal assistants could
become instant experts on any documented topic and answer questions in a faster,
easier to understand language.

Using basic methods like looping through an ngrams algorithm
shows that it can be done for search engines, not necessarily read-in
documents. Because of this limitation, a read in chat bot will be severely
limited in understanding of natural language without explicit training,
configuration, and scripting. Ultimately the approach that was taken in this
project is better suited for B2B products that have the resources needed to
maintain complex databases. A B2C product would require more automation at the
expense of the user tediously training the model.



### About The Library
#### ProcessArticle
Much of this project was based on the idea that the product would be fully automated. And so, the library holds the functions and classes needed to recreate the project in a different setting. The data gathering and cleaning is wrapped in the ProcessArticle class. This creates pre-cleaned objects that are ready for machine learning and analysis.

#### ChatBotModel
The model is a Frankenstein neural network using the looping ngrams and Word2Vec algorithms to measure the user article's relevancy against the read article. This is then tested against the gate parameter to return a true or false classification. Like with many NLP models, accuracy requires vast amounts of training data. The looping ngrams model can be trained in a reasonable amount of time, but the Word2Vec vectors require a lot of power to be effective. For this reason I have used Google's Word2Vec pre-trained model.

### Prototype
The prototype uses parameters generated from the random search test ran on Google's Compute Engine. Unfortunately, the quality of data provided for the test was too sparse for a perfect result, and I needed to run other tests to find the best parameters for this prototype.

### Running The Prototype
1. First set up your environment with the required packages. Package requirements
and versions are listed at the bottom of this readme. Please note that
additional NLTK packages may be required. Also, [Google's pretrained word2vec](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download) Will need to be
downloaded and unzipped into the "models" directory. The model is ~3GB unzipped
and is too large to push to GitHub. Alternatively, you can specify a train document
in the chat_bot_prototype.py file, however this is bound to decrease performance without
properly trained vectors.
2. Run the 'chat_bot_prototype.py' in your python interpreter. For Linux, the
process involves moving your terminal to the project directory and calling
python with chat_bot_prototype.py as an argument. Like so:
> $python3 chat_bot_prototype.py
3. There will be a few moments of load time as the chatbot loads in the google vectors.
4. The chatbot will supply a greeting and further instructions when the load is complete.
Your next input should be a sentence that you would like to know is true or false.
The chat bot will tell you based on the read-in article if the fact is true or
false. By default, the chatbot will read in the rules of chess.
5. This behavior will loop until the user inputs 'bye'. The bot will say Goodbye
and close. 

The purpose of this bot was for presentation purposes only and there are known limitations, such as:
- Single word user docs will throw an error. ( Stop words don't count )
- Quantity statements carry very little value and the model will not differentiate between them when passing the gate parameter.

### Variable Jargon
*Throughout this project I created variables names that associate to classes
and datatypes.*
- cd_data :Path as string to the data directory.
- cd_figures :  Path as string to the figures directory.
- cd_models :  Path to the models directory.
- gate : The threshold of proper relevance. A low gate will cause more false positives and a high gate will cause more false negatives.
- vector_scope :  An integer that defines the number of vectors to be counted in the Word2Vec model. This is an alias for the Word2Vec model's topn parameter.
- weight_mod :  A float number that is used as an exponent in the ngrams  loop where n**weight_mod = relevance_score.
- window : Parameter passed down to the Word2Vec model selecting the range of neighbors calculated around the selected number.

- epochs : Parameter passed down to the Word2Vec.train() method. This determines the number of times the documents will be looped over in the
 training process.

- vector_weight : The float number that the counter of vectors is multiplied by to determine what the vector weight is worth in the classification.
- user_article : ProcessArticle object from the user_doc.
- read_article : ProcessArticle object from which the model is reading from.
- train_article : ProcessArticle object from which the model is trained on.
- train_article_name : Title of the article that the model will train on.
- This variable is used for downloading the artilce from Wikipedia and saving the article to a file.
- doc : A raw, unprocessed string that has just been imported.
- article : A ProcessArticle object created from a doc.

### Sandbox
The Sandbox directory is for experiments, notes, and old code that is either not
used in the project or was created as a rough draft before applying it to the
completed notebook. These scripts are saved here with no intention of saving
dependencies or writing comments. Enter at your own risk.

### Directory Tree

```
.
├── chat_bot_prototype.py
├── data
│   ├── chatbot.db
│   ├── dummy_train.txt
│   ├── model_metrics.csv
│   ├── model_metrics_data.csv
│   ├── param_test_5x5(ModelMetrics).csv
│   ├── param_test_fullx100.csv
│   ├── param_test_fullx10.csv
│   ├── param_test_fullx50(ModelMetrics).csv
│   ├── rules of chess.txt
│   ├── sample_test.csv
│   ├── sample_test(ModelMetrics).csv
│   ├── stop_words.txt
│   ├── test_data.csv
│   ├── The_Grand_Mansion_Gate.txt
│   ├── tokens_1h.csv
│   ├── tokens_counts.csv
│   ├── tokens.csv
│   ├── top_1000_wiki.csv
│   └── train_data.txt
├── docs
│   ├── Google_Cloud_model_metrics.html
│   ├── Google_Cloud_model_metrics.xlsx
│   └── problem_statement.md
├── figures
│   ├── frequency-of-words-in-sentence.png
│   ├── most-frequent-words-in-sentence.png
│   ├── top-10-most-frequent-words.png
│   └── wordcloud.png
├── gather_articles.py
├── generate_test_data.py
├── library.py
├── models
│   ├── cloud_parameters.csv
│   ├── GoogleNews-vectors-negative300.bin
│   ├── GoogleNews-vectors-negative300.bin.gz
│   ├── parameters.csv
│   └── vectors.w2v
├── notebook.ipynb
├── notebook.py
├── params_query.sql
├── __pycache__
│   ├── library.cpython-37.pyc
│   ├── model.cpython-37.pyc
│   └── models.cpython-37.pyc
├── readme.md
├── sandbox
│   ├── confusion_matrix.py
│   ├── custom.py
│   ├── debug_evaluate_model.py
│   ├── gather_wiki_data.py
│   ├── gensim.py
│   ├── learning_wikipedia_API.py
│   ├── leftovers.py
│   ├── load_in_wordvectors.py
│   ├── models.py
│   ├── model_training.py
│   ├── Notes.md
│   ├── pseudo-code.md
│   ├── sandbox.py
│   ├── save_model_testing.py
│   ├── SentenceFrequency.py
│   ├── spacy.py
│   └── words2Vec.py
├── sources
│   └── articles.md
└── tree.txt

7 directories, 61 files
```

### Environment
```
numpy==1.18.1
matplotlib==3.1.3
pandas==1.0.1
wordcloud==1.8.0
gensim==3.8.3
wikipedia==1.4.0
nltk==3.4.5 ( additional downloads required )
tqdm==4.42.1




compiler   : GCC 7.3.0
system     : Linux
release    : 5.4.0-7642-generic
machine    : x86_64
processor  : x86_64
CPU cores  : 8
interpreter: 64bit
```
