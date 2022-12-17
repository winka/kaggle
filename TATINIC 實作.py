# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:19:12 2022

@author: user
"""
import pandas as pd 
import numpy as np 
import re
from sklearn import tree
from sklearn.model_selection import train_test_split 


train = pd.read_csv('C:\\Users\\user\\Downloads\\train.csv') 
test = pd.read_csv('C:\\Users\\user\\Downloads\\test.csv')


#觀察資料 step0   DAY1


#資料預處理  step1 cabin(艙位) embarked(要去的地方) 是沒有用的欄位 先去除 
train.drop(['Embarked','Cabin','PassengerId'],axis=1,inplace=True)
test.drop(['Embarked','Cabin','PassengerId'],axis=1,inplace=True)

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



