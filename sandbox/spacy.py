import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(chess)

for token in doc:
    print('{} : {}'.format(token, token.vector[:3]))
    
