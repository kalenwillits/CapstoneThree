read_article = ProcessArticle(doc)

#Gather user input
user_doc = 'Checkers are not a good'

#process user input

user_article = ProcessArticle(user_doc)


model = Model(user_article, read_article, gate=16.4)

model.response


ls = [1,2,3]

1 in ls


d1 = {'a':1, 'b':2}
d2 = {'c':3, 'd':4}

np.array(list(d1.values())) - np.array(list(d2.values()))
