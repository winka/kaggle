# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:19:12 2022

@author: winka
"""
import pandas as pd 
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score 
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
print("train") 
print(train.describe())
print(train.info())
print(train.isnull().sum())



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



# 數值類  類別類
train_num = []
train_cate = []

for dtype, feature in zip(train.dtypes, train.columns):
    if dtype == 'object':
        train_cate.append(feature)    
    else :
        train_num.append(feature)


# print(train_cate)

# plot 
# for i in train[train_num].columns:
#     plt.hist(train[i])
#     plt.title(i)
#     plt.show()







# items() turn dataframe in to items 
# Ex: train.groupby('title')['Age'].items() 
# output train['title'] then output train['Age']
    

# Goal column Fare fillna with column Parch median
# nan_fare = train.loc[df['Fare'].isnull(),'Pclass'].values[0]
# pclass_nanfare = train.loc[df['Pclass']==nan_fare,'Fare'].median()
# train['Fare'].fillna(pclass_nanfare,inplace=True)

# # Goal: column Fare 0 with column Pclass median
# zero_fare = df.loc[df['Fare']==0,'Pclass'].values[:]
# pclass_zerofare = df.loc[df['Pclass']==i,'Fare'].median()
# df.loc[df.Fare == 0, 'Fare'] = pclass_zerofare




# # Goal column Cabin fillna with column ?? 
# train['Cabin_firstword'] = train['Cabin'].str.extract('([A-Z])')

# print(pd.pivot_table(train,index='Cabin_firstword',values=['Fare','Pclass'])) 
# print(train['Cabin_firstword'].value_counts(normalize=True))

# print(df[(df['Pclass'] == 1) & (df['Cabin_firstword']=='A')]['Fare'].min())
# print(df[(df['Pclass'] == 1) & (df['Cabin_firstword']=='A')]['Fare'].max())

# print(df[(df['Pclass'] == 1) & (df['Cabin_firstword']=='B')]['Fare'].min())
# print(df[(df['Pclass'] == 1) & (df['Cabin_firstword']=='B')]['Fare'].max())

# print(df[(df['Pclass'] == 1) & (df['Cabin_firstword']=='C')]['Fare'].min())
# print(df[(df['Pclass'] == 1) & (df['Cabin_firstword']=='C')]['Fare'].max())




  





# .hist .boxplot etc

# #Result 除了 age 有常態分佈 其他都沒有
# # missing data count
# print(train.isna().sum())
# #Result column age, cabin, embarked have null (177,687,2)

# # corr 
# print(train_num.corr())
# sns.heatmap(train_num.corr())

# # columns age pclass sibsp vs survived 
# df_pivtable = pd.pivot_table(df, index='Survived', values=['Pclass','Age','SibSp'])
# print(df_pivtable)



#資料預處理  step3  pd,get_dummies  string -> int   DAY2
#模型無法用STRING 要先用LABELENCODER或ONE HOT ENCONDER 轉化STRING資料
#https://stackoverflow.com/questions/30384995/randomforestclassfier-fit-valueerror-could-not-convert-string-to-float

# remove Name 
# train_cate.remove('Name')

#Train & Score
train_catedata = pd.get_dummies(train[train_cate]).reset_index()
train_numdata = train[train_num].reset_index()
Train = pd.merge(train_catedata,train_numdata,on='index',how='right')

#Train & Score
y = Train['Survived']
X = Train.drop(['Survived'], axis=1)


#step  3 決策樹測試 
X_train, X_test, y_train, y_test = train_test_split(X,y)
tree = DecisionTreeClassifier()
iris_clf = tree.fit(X_train, y_train)

print(cross_val_score(tree, X_test, y_test, cv=10).mean())
print(iris_clf.score(X_test,y_test))

# 0.7723320158102767
# 0.8340807174887892
      






#優化分數 預處理加強?  選擇其他演算法? 變更其目前演算法中的參數? DAY3



#print(train.isna().sum())


# 獨熱編碼 + 羅吉斯迴歸 模型驗證 交叉驗證
# df_temp = pd.DataFrame()
# df_temp = pd.get_dummies(df)
# train_X = df_temp[:train_num]
# emstimator = LogisticRegression()

# print(f'cross_validation:{cross_val_score(emstimator,train_X,train_Y,cv=5).mean()}')#
# #模型驗證(model validation)
# #交叉驗證(cross_val_score)
# #每次從全部資料取不同的部分分別擔任train與test 進行多次運算(運算數字根據cv的數字而定) 
# #彌補了train_test_split只計算一次的缺點(holdout set)


