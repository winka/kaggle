"""
Created on Wed Sep 21 16:59:02 2022

@author: user
"""
import numpy as np 
import pandas as pd 

# read file train test
train = pd.read_csv('C:\\Users\\user\\Downloads\\train.csv') 
test = pd.read_csv('C:\\Users\\user\\Downloads\\test.csv')
# describe train test
train_describe = train.describe()
test_describe = test.describe()

# train test is na 
train_isna = train.isna().sum() #count不行 count使用於 全部項次相加
test_isna = test.isna().sum() #sum用於各項次加總


#test train concat (依照columns項目來將兩個表格合併)
combine = pd.concat([train,test]) 
#print(combine.isnull().sum())

# combin 丟掉 cabin embarked 
combine =  combine.drop(['Cabin','Embarked'], axis = 1)

#dataframe.values[0] 輸出dataframe 且 index = 1 的值
pclass = combine.loc[combine['Fare'].isnull(),'Pclass'].values[0]


#print(combine.iloc[1043]) # iloc查詢資料 = 原本列數-1

#印出 combine Pclass ==pclass 的 Fare 欄位 的中位數
median_fare = combine.loc[combine.Pclass==pclass,'Fare'].median()

combine.loc[combine.Fare.isnull(),'Fare'] = median_fare



# 正規式表示法 https://www.kaggle.com/code/dennisbakhuis/titanic-k-nearest-neighbor-knn-frmscratch-0-813/notebook
#取A-Z a-z 無限次 直到換行符號 或沒有值可以取
# + 取前面的正規式多次或一次
# \ 转义特殊字符（允许你匹配 '*', '?', 或者此类其他），或者表示一个特殊序列；特殊序列之后进行讨论。 

combine['title'] = combine['Name'].str.extract('([A-Za-z]+)\.', expand=True)
combine['title'].unique()



#print(type(combine.groupby('title')))
#print(combine.Age.isnull().value_counts())
               #                     
for title ,  age in combine.groupby('title')['Age'].median().items():
    #print(title, age)
    combine.loc[(combine['title'] ==title) & (combine['Age'].isnull()),'Age'] = age
     

print(combine['Age'])






