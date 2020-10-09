    # TN = true negative
    # FP = false positive
    # FN = false negative
    # TP = true positive


    # Accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
    # Recall = ( TP ) / ( TP + FN )
    # Precision = ( TP ) / ( TP + FP )
    # False Positive Rate = ( FP ) / ( TN + FP )

# Running notebook kernel first.
from tqdm import tqdm

test_name = 'test.txt'



for user_doc in tqdm(test_df['user_doc']):
    user_article = ProcessArticle(user_doc)
    test_model = ChatBotModel(user_article=user_article,
                        read_article=chess,
                        train_article=chess,
                        train_article_name='train_sample.txt',
                        gate=30,
                        weight_mod=1.5,
                        window=50,
                        epochs=15,
                        vector_weight=10,
                        vector_scope=5)
    idx = test_df[test_df['user_doc'] == user_doc].index[0]
    test_df.loc[idx, 'predict'] = test_model.prediction[0]
    test_df.loc[idx, 'score'] = test_model.prediction[1]


nulls = test_df.isnull().sum()

def confusion_matrix(test_df=test_df, cd_data=cd_data):
    """
    calculates the confusion_matrix from 'type' and 'predict' columns in a test
    pandas dataframe.
    """
    test_df = pd.read_csv(cd_data+'test_data.csv')

    true_negative = len(test_df[(test_df['type'] == False)
                        &
                        (test_df['predict'] == False)])

    false_positive = len(test_df[(test_df['type'] == False)
                        &
                        (test_df['predict'] == True)])

    false_negative = len(test_df[(test_df['type'] == True)
                        &
                        (test_df['predict'] == False)])

    true_positive = len(test_df[(test_df['type'] == True)
                        &
                        (test_df['predict'] == True)])

    confusion_matrix = {'TN': true_negative, 'FP': false_positive, 'FN': false_negative, 'TP': true_positive}

    return confusion_matrix

    confusion_matrix(test_df=test_df)
