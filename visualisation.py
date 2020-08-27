# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:45:43 2020

@author: Alok Garg
"""
import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def CategoricalVariableBarplot(df,feature_list,target = 'Loan_Status'):
    for feature in feature_list:
        g = pd.crosstab(df[feature],df[target])
        g.div(g.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = True,figsize=(12,8))
        # sns.barplot(x = feature,data=df)
        # df[feature].value_counts().plot.bar()
        if not os.path.exists('./images'):
            os.mkdir('./images')
        if not os.path.exists(os.path.join('./images','barplots')):
            os.mkdir(os.path.join('./images','barplots'))
        plt.savefig('./images/barplots/{}'.format(feature))

        # plt.show()
        # return g
def CountValues(df,feature_list):
    for feature in feature_list:
        df[feature].value_counts().plot(kind= 'bar',figsize = (12,8))
        if not os.path.exists('./images'):
            os.mkdir('./images')
        if not os.path.exists(os.path.join('./images','ValueCounts')):
            os.mkdir(os.path.join('./images','ValueCounts'))
        plt.savefig('./images/ValueCounts/{}'.format(feature))

def NumericalVariableBoxPlot(df,feature_list ):
    plt.figure(figsize = (20,8))
    for feature in feature_list:

        s = sns.boxplot(df[feature],data = df,orient='v')
        if not os.path.exists('./images'):
            os.mkdir('./images')
        if not os.path.exists(os.path.join('./images','Boxplot')):
            os.mkdir(os.path.join('./images','Boxplot'))
        s.figure.savefig('./images/Boxplot/{}'.format(feature))



def NumericalVariableDistplot(df,feature_list):
    for feature in feature_list:
        s = sns.distplot(df[feature],hist=True,kde = True,color= 'g')
        if not os.path.exists('./images'):
            os.mkdir('./images')
        if not os.path.exists(os.path.join('./images','Distplot')):
            os.mkdir(os.path.join('./images','Distplot'))
        s.figure.savefig('./images/Distplot/{}'.format(feature))






if __name__ == '__main__':
    print('visualizing the data')
    train = pd.read_csv('./dataset/train.csv')
    test = pd.read_csv('./dataset/test.csv')
    print('univariate analysis')
    # 1. Gender ---> count value apn plot the bar plot for gender
    feature_names = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area', 'Loan_Status','Credit_History']
    numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
    # CategoricalVariableBarplot(train,feature_names,target = 'Loan_Status')
    # CountValues(train,feature_names)
    # NumericalVariableBoxPlot(train,feature_list = numerical_features)
    NumericalVariableDistplot(train,numerical_features)
    sns.boxplot(x = train['CoapplicantIncome'],data = train,orient='v')

