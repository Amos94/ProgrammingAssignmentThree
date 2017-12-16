import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from random import shuffle


data_rows = list()
feature_name = list()

'''
Parsing arff file
'''
with open('relevant_resources/weka_institutions.arff', 'r') as f:
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

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

clf = LogisticRegression()
scores = sklearn.model_selection.cross_val_score(clf, X, Y, cv=10)
print(scores.mean())
