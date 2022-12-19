## Machine Learning Step

1. Understand the shape of Data </br>
      1-1 info .describe etc 探索資料資訊</br>
      1-2.hist .boxplot etc </br>
      1-3 corr相關係數 </br>
      1-4 value counts 
1. Data Cleaning</br>
     2-1 missing data count 填補缺失值
1. Data exploration
1. Feature Engineering
1. Data preprocessing for Model
1. Basic Model Building
1. Model Tunging
1. Ensemble Model Building
1. Result
  
### 實作過程觀察到的訊息
首先用 describe 來看資料相關資訊
```python
    print(df.describe())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20describe.PNG?raw=true)

### 再來用 info 來看資料相關資訊
```python
    print(df.info())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20info.PNG?raw=true)

### 數值型資料相關係數
```python
   sns.heatmap(df.corr())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20corr.png?raw=true)


## 我的觀察

### 1. Age, Cabin 欄位有缺失值需要再填補缺失值  
### 2. Data中有 數值型 與 文字型 兩種資料

## 經過觀察後對資料處理
### 1. 從Name欄位中擷取稱謂(Mr, Mrs)的中位數來填補 Age 缺失值


 1. 數值型欄位 .hist   
```python
for i in df[train_num].columns:
    plt.hist(df[i])
    plt.title(i)
    plt.show()
```

| Columns name  | Columns plot  |
| ------------- |:-------------:|
| Survived           | ![image](https://github.com/winka/IMG/blob/main/tantic%20survived.png?raw=true)          |
Parch      | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20parch.png?raw=true)             |
| Sibsp           | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20sibsp.png?raw=true)          |
|  Fare     | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20fare.png?raw=true)            |
| Age      |![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20age.png?raw=true)              |
| Parch      |![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20parch.png?raw=true)              |
| Pclass      |![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20pclass.png?raw=true)              |

## 我的觀察
### Fare SibSp Fare 欄位有離群值得發生  

## 使用稱謂(Mr, Mrs)的中位數來填補Age欄位缺失值
```
# Goal column:Age fillna 
# Step1 column:Name extract mr mrs sir etc
train['title'] = train['Name'].str.extract('([A-Za-z]+)\.')

# Step2 mr mrs sir age median  
title_Agemean = pd.pivot_table(train, index='title', values='Age')
# Step3 column:age fillna with column:title age median 
for i, j  in df.groupby('title')['Age'].median().items():
    train.loc[(train['title'] == i) & (train['Age'].isnull()) ,'Age'] = j
print(train['title'].value_counts())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20table%20title.PNG?raw=true)

```
#Train & Score
y = Train['Survived']
X = Train.drop(['Survived'], axis=1)


#step  3 決策樹測試 
X_train, X_test, y_train, y_test = train_test_split(X,y)
tree = DecisionTreeClassifier()
iris_clf = tree.fit(X_train, y_train)

print(cross_val_score(tree, X_test, y_test, cv=10).mean())
print(iris_clf.score(X_test,y_test))
```
### 結果；目前為77%
![image](https://github.com/winka/IMG/blob/main/tantic%20score.PNG?raw=true)


## 我的觀察

### 1. 除了填補缺失值以外是否還有其他方法能強化正確率(目前為77%)

## 經過觀察後對資料處理
### 1. 對資料中的兩個欄位(Sex, Fare)做群聚編碼以求改善正確率
### 2. 使用GridSearchCV來測試同樣一個Data在不同模型下的表現
## 1. 對資料中的兩個欄位(Sex, Fare)做群聚編碼以求改善正確率
```
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
```
### 結果；針對Data特徵使用群聚編碼後再同一個模型下正確率有明顯提升(沒有用群聚編碼 77% -> 有使用群聚編碼 83%)
![image](https://github.com/winka/IMG/blob/main/tantic%20%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC%E5%BE%8Cscore.PNG?raw=true)

## 2. 使用GridSearchCV來測試同樣一個Data在不同模型下的表現
```
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
```
### 結果；GridSearchCV選擇使用RandomForestClassifier來作為我的模型，但Data在該模型的正確率下降(使用DecisionTreeClassifier 83% -> 使用RandomForestClassifier 80%)
![image](https://github.com/winka/IMG/blob/main/tantic%20gridsearchcv%20score.PNG?raw=true)
##  總結
### 1.在既有的特徵中加入額外的特徵能夠使大幅提高正確率(77% -> 83%)，應該在尋找資料中更多的關聯性組合出更多額外特徵
### 2.目前單純選擇模型無法為我的資料提高正確率，或許可以加入在各個模型加入他們個別的參數使正確率提高










