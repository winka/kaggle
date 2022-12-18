## Machine Learning Step

1. Understand the shape of Data </br>
      1-1 info .describe etc 資料相關資訊</br>
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
