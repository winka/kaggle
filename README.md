## Tantic Machine Learning 

希望能透過這次機器學習模型製作的過程，發現甚麼樣身分的鐵達尼克號乘客(例如:是否購買較貴的船艙，性別為男性或是女性等)
能逃離這場災難，以及讓我的機器模型能夠正確預測乘客是否生存

### Tantic Machine Learning outline
* 資料探索
* 特徵值處理
* 模型訓練 & 結果
* 模型預測正確率強化
* 總結

# 資料探索
## 鐵達尼號資料表欄位
![image](https://github.com/winka/IMG/blob/main/data%20dictionary.PNG?raw=true)
### 資料來源:https://www.kaggle.com/competitions/titanic/overview

### 實作過程觀察到的訊息
用 describe 來觀察資料的平均數、中位數、最小值以及最大值等訊息
```python
    print(df.describe())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20describe.PNG?raw=true)

用 info 查看各個欄位有無缺失值，以及各自的資料型態
```python
    print(df.info())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20info.PNG?raw=true)

數值型資料相關係數
```python
   sns.heatmap(df.corr())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20corr.png?raw=true)

數值型欄位直方圖   
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
1. Age 欄位有缺失值需要再填補缺失值  
2. Data中有 數值型 與 文字型 兩種資料
3. Fare SibSp Fare 欄位有離群的得發生
4. 從資料Corr中可以發現Survived與Fare欄位有正關係，Fare欄位的值越高越有可能活下來，反之亦然
5. 從資料Corr中可以發現Survived與Pclass欄位有負關係，Pclass欄位的值越低越有機會存活，反之亦然


# 特徵值處理
### 1. 從Name欄位中擷取稱謂(Mr, Mrs)的中位數來填補 Age 缺失值

使用稱謂(Mr, Mrs)的中位數來填補Age欄位缺失值
```python
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

# 模型訓練 & 結果
```python
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
### 結果:77%
![image](https://github.com/winka/IMG/blob/main/tantic%20score.PNG?raw=true)

## 我的觀察
1. 目前正確判斷率為不理想的77%，希望藉由增加額外的特徵讓模型有更多判斷的依據、或是換一個機器學習模型來強化正確率

# 模型預測正確率強化
### 1. 對資料中的兩個欄位(Sex, Fare)做群聚編碼增加額外的特徵強化正確率
### 2. 使用GridSearchCV來測試同樣一個Data在不同模型下的表現

## 對資料中的兩個欄位(Sex, Fare)做群聚編碼強化正確率
```python
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
### 結果:針對Data特徵使用群聚編碼後再同一個模型下正確率有明顯提升(沒有用群聚編碼 77% -> 有使用群聚編碼 83%)
![image](https://github.com/winka/IMG/blob/main/tantic%20%E7%BE%A4%E8%81%9A%E7%B7%A8%E7%A2%BC%E5%BE%8Cscore.PNG?raw=true)

## 使用GridSearchCV來測試同樣一個Data在不同模型下的表現
```python
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
### 結果:GridSearchCV選擇使用RandomForestClassifier來作為我的模型，但資料在該模型的正確率下降(使用DecisionTreeClassifier 83% -> 使用RandomForestClassifier 80%)
![image](https://github.com/winka/IMG/blob/main/tantic%20gridsearchcv%20score.PNG?raw=true)
#  總結
### 1.在既有的特徵中加入額外的特徵能夠使大幅提高正確率(77% -> 83%)。應該在資料中尋找更多的關聯性並從中組合出更多額外特徵
### 2.目前只使用模型選擇無法提高正確率反而會使正確率降低，未來可以在各個模型加入他們個別的參數，並進行調整讓模型正確率提高










