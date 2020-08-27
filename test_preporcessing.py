# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:53:02 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import DataPreprocessing as dp







if __name__ == '__main__':
    print('preprocessing test dataset')
    test = pd.read_csv('./dataset/test.csv')
    # test.drop(test[test['ApplicantIncome'] == 0].index, inplace = True)
    test_1 = dp.FillMissingValues(test)
    test_2 = dp.DataTransFormation(test_1)
    encode_feature = ['Gender', 'Married','Education','Self_Employed','Property_Area']
    test_3 = dp.lableEncoding(test_2,encode_feature)
    # test_4 = dp.ConvertTONormal(test_3)
    test_5 = dp.standrize(test_3)
    preprocessed_test = test_5.iloc[:,1:]
    preprocessed_test.to_csv('./dataset/preprocessed_test.csv',index = False)
