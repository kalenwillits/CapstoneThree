ead_article=read_article
train_article=train_article
train_article_name=train_article_name
test_df=test_df
test_name=test_name
parameters=model.parameters
cd_data=cd_data
parameters

user_article=user_article
                    read_article=chess
                    train_article=chess
                    train_article_name='train_sample.txt'
                    gate=20
                    weight_mod=1.5
                    window=50
                    epochs=15
                    vector_weight=10
                    vector_scope=5

# change paramters in model to match dict.
for user_doc in tqdm(test_df['user_doc']):
    user_article = ProcessArticle(user_doc)
    test_model = ChatBotModel(user_article=user_article,
                        read_article=read_article,
                        train_article=train_article,
                        train_article_name=train_article_name,
                        gate=parameters['gate'],
                        weight_mod=parameters['weight_mod'],
                        window=parameters['window'],
                        epochs=parameters['epochs'],
                        vector_weight=parameters['vector_weight'],
                        vector_scope=parameters['vector_scope'])
    idx = test_df[test_df['user_doc'] == user_doc].index[0]
    test_df.loc[idx, 'predict'] = test_model.prediction[0]
    test_df.loc[idx, 'score'] = test_model.prediction[1]

    test_df.to_csv(cd_data+test_name+'.csv')

    # Write model metrics, params, and performance to file.
    metrics = ModelMetrics(test_df=test_df)


    # initializing DataFrame.
    metrics_dict = {}
    for key in parameters.keys():
        metrics_dict[key] = []

    for key in metrics.matrix.keys():
        metrics_dict[key] = []

    metrics_columns = ['accuracy',
                        'precision',
                        'recall',
                        'false_positive_rate']

    for column in metrics_columns:
        metrics_dict[column] = []

    metrics_dict['gate'].append(test_model.gate)
    metrics_dict['weight_mod'].append(test_model.weight_mod)
    metrics_dict['window'].append(test_model.window)
    metrics_dict['epochs'].append(test_model.epochs)
    metrics_dict['vector_scope'].append(test_model.vector_scope)
    metrics_dict['vector_weight'].append(test_model.vector_weight)
    metrics_dict['TN'].append(metrics.matrix['TN'])
    metrics_dict['FP'].append(metrics.matrix['FP'])
    metrics_dict['FN'].append(metrics.matrix['FN'])
    metrics_dict['TP'].append(metrics.matrix['TP'])
    metrics_dict['accuracy'].append(metrics.accuracy)
    metrics_dict['precision'].append(metrics.precision)
    metrics_dict['recall'].append(metrics.recall)
    metrics_dict['false_positive_rate'].append(metrics.false_positive_rate)

metrics_dict
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(cd_data+test_name+'(ModelMetrics).csv')
