# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:19:09 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import os


def predict_lr(test_data,model):
    predictions = model.predict(test_data)
    return np.array(predictions)








if __name__ == '__main__':
    print('making the submission file')
    sample_submission = pd.read_csv('./dataset/sample_submission.csv')

    # sample_submission.drop(sample_submission[(sample_submission['Loan_ID'] == 'LP001153')].index, inplace = True)
    # sample_submission.drop(sample_submission[(sample_submission['Loan_ID'] == 'LP001607')].index, inplace = True)
    processed_test = pd.read_csv('./dataset/preprocessed_test.csv')
    model = joblib.load('./models/model_rf.pkl')
    predicted_value = predict_lr(processed_test,model)

    sample_submission['Loan_Status'] = predicted_value