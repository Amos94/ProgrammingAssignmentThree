import sys
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from random import shuffle


data_rows = list()
feature_name = list()

'''
Parsing arff file
'''
with open(sys.argv[1], 'r') as f:
    raw_data_rows = f.read()
    arff_header = raw_data_rows[raw_data_rows.find('@ATTRIBUTE'):raw_data_rows.find('@DATA') - 2]
    arff_header= arff_header.split('\n')
    feature_names = list()
    for column_title in arff_header:
        column_title = column_title.lstrip('@ATTRIBUTE ')
        column_title = column_title[:column_title.find(' ')]
        feature_names.append(column_title)
    raw_data_rows = raw_data_rows[raw_data_rows.find('@DATA') + 6:]
    raw_data_rows = raw_data_rows.split('\n')
    for raw_data_row in raw_data_rows:
        row = raw_data_row.split(',')
        row_processed = list()
        for element in row[:-1]:
           row_processed.append(int(element))
        if row[-1] == 'True':
            row_processed.append(True)
        else:
            row_processed.append(False)
        data_rows.append(row_processed)

shuffle(data_rows)

data = pd.DataFrame(data_rows)#converting into panda data frame
data.columns = feature_names#adding column titles (every attribute from the arff file can now be accessed trough the data dictionary)

X = data.drop('class', axis=1)#features (a.k.a. "X")
Y = data['class']#labels (a.k.a. "Y")

overall_accuracy = 0
for i in range(10):
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
    clf = LogisticRegression()
    clf.fit(X_train, Y_train)
    #inspiration from: http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    y_true = Y_test
    y_pred = clf.predict(X_test)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    print('Confusion matrix for fold', i + 1)
    print(tp, '\t', fn, '\n', fp, '\t', tn)
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    print('Accuracy:', str(round((accuracy * 100),3)) + '%') 
    print(100 * '*')
    overall_accuracy += accuracy
    
print('Average accuracy:', str(round((overall_accuracy*10),3)) + '%')
    

