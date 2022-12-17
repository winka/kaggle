# -*- coding: utf-8 -*-
"""
@author: winka
Created on Wed Oct 26 11:19:12 2022
Code Name: T VER 1.002
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import re
from sklearn import tree

from sklearn.model_selection import train_test_split 

# Machine Learning Step
##---------------------------------------------##
# understand the shape of data 了解資料的結構
    # .info .describe etc 
    # .hist .boxplot etc
    # missing data count
    # corr 
# Data cleaning
    # value counts
    # missing data
    # pd.get_dummies 
    # feature scaling
        # 避免有些值過大而影響較小的值
    # normalizing
        # 通常的值都為常態分佈
# Data exploration 
# Feature Engineering
# Data preprocessing for Model
# Basic Model Building
# Model Tunging 
# Ensemble Model Building 
# Result

train = pd.read_csv('C:\\Users\\user\\Downloads\\train.csv') 
test = pd.read_csv('C:\\Users\\user\\Downloads\\test.csv')

# understand the shape of data 

    # .info .describe etc 
print("Train Describe Info")
print(train.describe())
print(train.info()) 
# Result
# train data[age, cabin ,embarked] have nan

# .hist .boxplot etc
train_numberic = train.loc[:,['Survived','Pclass','SibSp','Parch','Fare']]
#train_category = train.loc[]
#print(train_numberic[0])
plt.figure(figsize=(10,10)) 
for i in train_numberic.columns:
    # hist
    plt.hist(train_numberic[i])
    plt.title(i)
    plt.show()
    







 
'''
#資料預處理  step2 遺漏值處理 
# 抓出name 欄位中的稱謂 如 mr sir mrs master 等  
train['title'] = train['Name'].str.extract('([A-Za-z]+)\.')  
#test['title'] = train['Name'].str.extract('([A-Za-z]+)\.')
# 從各個稱謂(titl e) 中做平均加入 age
for title, age in train.groupby('title')['Age'].median().items():
    train.loc[(train['title'] == title) & (train['Age'].isnull()) ,'Age'] = age


#資料預處理  step3  pd,get_dummies  string -> int   DAY2
#模型無法用STRING 要先用LABELENCODER或ONE HOT ENCONDER 轉化STRING資料
#https://stackoverflow.com/questions/30384995/randomforestclassfier-fit-valueerror-could-not-convert-string-to-float
train_getdummies = pd.get_dummies(train['Sex'])


#插入 train'sex' get_dummies的欄位
New_train = pd.concat([train, train_getdummies], axis=1)




#step  3 決策樹測試?        
New_train.drop(['Name','Sex','Ticket','title'],axis=1,inplace=True)

X = New_train[['Pclass','Age','SibSp','Parch','Fare','female','male']]
y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X,y)
tree = tree.DecisionTreeClassifier()
tree.fit(X_train,y_train)


print(tree.score(X_test,y_test))

#優化分數 預處理加強?  選擇其他演算法? 變更其目前演算法中的參數? DAY3



#print(train.isna().sum())
'''


