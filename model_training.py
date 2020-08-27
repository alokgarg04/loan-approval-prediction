# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:30:14 2020

@author: Alok Garg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

def logistic_regression(feature,target):
    x_train,x_test,y_train,y_test = train_test_split(feature, target,test_size = 0.33,stratify=target,random_state=42)
    clf = LogisticRegression(verbose=4)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    score = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return clf,score,matrix

def random_forest(feature,target):
    x_train,x_test,y_train,y_test = train_test_split(feature, target,test_size = 0.33,stratify=target,random_state=42)
    clf = RandomForestClassifier(verbose=4)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    score = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return clf,score,matrix





if __name__ == '__main__':
    print('training the model')
    preprocessed_train = pd.read_csv('./dataset/preprocessed_train.csv')
    y = preprocessed_train.iloc[:,-1]
    x = preprocessed_train.iloc[:,:-1]
    # clf,score,matrix = logistic_regression(x,y)
    clf_rf,score_rf,matrix_rf = random_forest(x,y)
    if not os.path.exists('./models'):
        os.mkdir('./models')
    joblib.dump(clf_rf,'./models/model_rf.pkl')