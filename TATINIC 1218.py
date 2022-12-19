# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:19:12 2022

@author: winka
"""
import pandas as pd 
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score 
from sklearn.preprocessing import MinMaxScaler


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Machine Learning Step
##---------------------------------------------##
# understand the shape of data 了解資料的結構
    # .info .describe etc 
    # .hist .boxplot etc
    # missing data count
    # corr 
# Data cleaning
    # value counts 
    # missing data fix 
# Data exploration 
# Feature Engineering
# Data preprocessing for Model
# Basic Model Building
# Model Tunging 
# Ensemble Model Building 
# Result

train = pd.read_csv('D:\\My\\python\\kaggle\\titanic\\train.csv') 
test = pd.read_csv('D:\\My\\python\\kaggle\\titanic\\test.csv')


train = train.drop(['Cabin','Embarked'],axis=1)
#觀察資料 step0   DAY1
# .info .describe etc
# print("train") 
# print(train.describe())
# print(train.info())
# print(train.isnull().sum())

# Day2
# Goal column:Age fillna 
# Step1 column:Name extract mr mrs sir etc
train['title'] = train['Name'].str.extract('([A-Za-z]+)\.')

# Step2 mr mrs sir age median  
title_Agemean = pd.pivot_table(train, index='title', values='Age')
# Step3 column:age fillna with column:title age median 
for i, j  in train.groupby('title')['Age'].median().items():
    train.loc[(train['title'] == i) & (train['Age'].isnull()) ,'Age'] = j

print(train.title.value_counts())
train = train.drop('Name', axis=1)

# 取一個類別型欄位, 與一個數值型欄位, 做群聚編碼
# Columns Sex and fare Group by Encoding
# 取出一個類別型欄位對另一個數值型欄位做運算
# 找出兩個欄位的mean, max, min, median mode(眾數) 等生成特徵欄位
# 特徵欄位越多越好(通常) 
sexfare_mean = train.groupby('Sex')['Fare'].mean().reset_index()
sexfare_max = train.groupby('Sex')['Fare'].max().reset_index()
sexfare_min = train.groupby('Sex')['Fare'].min().reset_index()
sexfare_median = train.groupby('Sex')['Fare'].median().reset_index()
sexfare_mode = train.groupby('Sex')['Fare'].apply(lambda x: x.mode()[0]).reset_index()


# 合併到train
temp = pd.DataFrame()
temp = pd.merge(sexfare_mean, sexfare_max, on='Sex', how='left')
temp = pd.merge(temp, sexfare_min, on='Sex', how='left')
temp = pd.merge(temp, sexfare_median, on='Sex', how='left')
temp = pd.merge(temp, sexfare_mode, on='Sex', how='left')

temp.columns = ['Sex','sexfare_mean','sexfare_max','sexfare_min','sexfare_median','sexfare_mode']

train = pd.merge(train,temp,on='Sex',how='right')


# 數值型  類別型 分類
train_num = []
train_cate = []

for dtype, feature in zip(train.dtypes, train.columns):
    if dtype == 'object':
        train_cate.append(feature)    
    else :
        train_num.append(feature)


# print(train_cate)

# plot 
for i in train[train_num].columns:
    plt.hist(train[i])
    plt.title(i)
    plt.show()

# .hist .boxplot etc

# #Result 除了 age 有常態分佈 其他都沒有
# # missing data count
# print(train.isna().sum())
# #Result column age, cabin, embarked have null (177,687,2)

# # corr 
print(train_num.corr())
sns.heatmap(train_num.corr())

# # columns age pclass sibsp vs survived 
# df_pivtable = pd.pivot_table(df, index='Survived', values=['Pclass','Age','SibSp'])
# print(df_pivtable)


#Train & Score
train_catedata = pd.get_dummies(train[train_cate]).reset_index()
train_numdata = train[train_num].reset_index()
Train = pd.merge(train_catedata,train_numdata,on='index',how='right')

# Train & Score
y = Train['Survived']
X = Train.drop(['Survived'], axis=1)


# 分類成訓練集 X_train y_train以及測試集 X_test y_test
X_train, X_test, y_train, y_test = train_test_split(X,y)
tree = DecisionTreeClassifier()


# GridSearchCV 試驗
# GridSearchCV可以用來尋找這個data在哪個分類器以及參數下表現最好

# Step1 做3個分類器List
# 1. 分類器的名字(必要?)
classifier_names = [
    'LogisticRegression',
    'KNeighborsClassifier',
    'RandomForestClassifier',
    'DecisionTreeClassifier'
    ]
# 2. 分類器函數(必要?)
classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    DecisionTreeClassifier()
              ]
# 3. 分類器參數(必要)
parameters = [
    {},
    {},
    {},
    {}
]

# Step2 GridSearchCV 測試

# zip 用法
# >>> for i, j  in zip(x, y):
#         print(i,j)
# a 1
# b 2
# c 3
result = []
for name, classifier, params in zip(classifier_names,classifiers,parameters):
    gsearch = GridSearchCV(classifier,param_grid=params)
    fitted = gsearch.fit(X_train,y_train)
    y_pred = gsearch.predict(X_test)
    score = gsearch.score(X_test,y_test)
    
    result.append({
        'Name':name,
        'Model':gsearch,
        'Score':score,
        'Prediction': y_pred
        })
    
# result 排序
# 尋找最好的訓練模型與他的分數   
result.sort(key = lambda x: x['Score'], reverse = True)
best_model = result[0]['Model']
best_score = result[0]['Score']
best_model_name = result[0]['Name']

print(f'best_model: {best_model}')
print(f'best_score: {best_score}')











