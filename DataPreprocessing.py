# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:14:03 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def basic_details(df):
    print(df.shape)
    print()
    print(df.columns)
    print()
    print(df.dtypes) # 6 categorical featues,Credit_History should also be categorical variable



def categorical(df,feature_names):
    value_count_dict = {}
    for feature in feature_names:
        print(df[feature].value_counts())
        print('----------------------------------------------------')
        value_count_dict[feature] = df[feature].value_counts()
    return value_count_dict

def CheckNullValue(df):
    print(df.isnull().sum())
    null_value_df = pd.DataFrame(df.isnull().sum(),columns = ['number of null values'])
    return null_value_df

def Stats(df):
    stats = df.describe().T
    return stats

def FillMissingValues(df):
    def Gender(feature = "Gender"):
        df[feature] = df[feature].fillna(df[feature].mode()[0]) # mode()[0] gives value and mode() gives output as object
        return df

    def Married(feature = 'Married'):
        df[feature] = df[feature].fillna(df[feature].mode()[0])
        return df

    def Dependents(feature = 'Dependents'):
         df[feature] = df[feature].fillna(df[feature].mode()[0])
         return df
    def Self_Employed(feature = 'Self_Employed'):
        df[feature] = df[feature].fillna(df[feature].mode()[0])
        return df
    def Credit_History(feature = 'Credit_History'):
        df[feature] = df[feature].fillna(df[feature].mode()[0])
        return df
    def LoanAmount(feature = 'LoanAmount'):
        # Loan amout has mny outliers so will use meadin
        df[feature] = df[feature].fillna(df['LoanAmount'].median())
        return df
    def Loan_Amount_Term(feature = 'Loan_Amount_Term'):
        df[feature] = df[feature].fillna(df[feature].mode()[0])
        return df


    df = Gender(feature = "Gender")
    df = Gender(feature = "Married")
    df = Gender(feature = "Dependents")
    df = Self_Employed(feature = 'Self_Employed')
    df = Credit_History(feature = 'Credit_History')
    df = LoanAmount(feature = 'LoanAmount')
    df = Loan_Amount_Term(feature = 'Loan_Amount_Term')
    return df

def DataTransFormation(df):
    df['Dependents'] = df['Dependents'].apply(lambda x : 3 if x == '3+' else x)
    return df


def lableEncoding(df,features):
    label_encoder = LabelEncoder()
    updated_values = []
    for feature in features:
        encoded_values =   label_encoder.fit_transform(df[feature].values)
        # print('replace the value in dataset')
        df[feature] = encoded_values
    return df

def ConvertTONormal(df):
    df['ApplicantIncome'] = np.log(df['ApplicantIncome'])
    df['LoanAmount'] = np.log(df['LoanAmount'])
    return df

def standrize(df):
    std_scalar = MinMaxScaler()
    # std_scalar = StandardScaler()
    df['CoapplicantIncome'] =  std_scalar.fit_transform(df['CoapplicantIncome'].values.reshape(-1,1))
    df['Loan_Amount_Term'] =  std_scalar.fit_transform(df['Loan_Amount_Term'].values.reshape(-1,1))
    df['ApplicantIncome'] =  std_scalar.fit_transform(df['ApplicantIncome'].values.reshape(-1,1))
    df['LoanAmount'] =  std_scalar.fit_transform(df['LoanAmount'].values.reshape(-1,1))
    return df



if __name__ == '__main__':
    print('data prprocessing started')
    train = pd.read_csv('./dataset/train.csv')
    test = pd.read_csv('./dataset/test.csv')
    cols = train.columns
    basic_details(train)
    feature_names = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area', 'Loan_Status','Credit_History']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
    value_count_dict = categorical(train,feature_names)
    null_value_df = CheckNullValue(train)
    print("======================================")
    stats = Stats(train)
    ''' for categorical variable will use mode and for numerical will use mmean or median'''
    train_1 = FillMissingValues(train)
    null_value_df_updated = CheckNullValue(train_1)
    # in dependents column one value is 3+ remove that and
    train_2 = DataTransFormation(train_1)
    train_2['Dependents'].value_counts()

    ''' while traing the model will use fit() function which take only numerical value
        so convert all categorical values to numeric'''
    # we can male a function and apply but will use label encoder just to explore sklearn
    encode_feature = ['Gender', 'Married','Education','Self_Employed','Property_Area','Loan_Status']
    train_3 = lableEncoding(train_2,encode_feature)
    # train_3.to_csv('./dataset/train_3.csv',index = False)

    ''' we have seen that ApplicantIncome is left skewed  so log transformation can make it normal'''
    # train_4 = ConvertTONormal(train_3)
    # train_4.to_csv('./dataset/train_4.csv',index = False)

    train_5 = standrize(train_3)
    preprocessed_train = train_5.iloc[:,1:]
    preprocessed_train.to_csv('./dataset/preprocessed_train.csv',index = False)
