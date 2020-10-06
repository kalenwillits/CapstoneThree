read_article = ProcessArticle(doc)

#Gather user input
user_doc = 'Checkers are not a good'

#process user input

user_article = ProcessArticle(user_doc)


model = Model(user_article, read_article, gate=16.4)

model.response
