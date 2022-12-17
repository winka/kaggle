### Machine Learning Step

1. Understand the shape of Data
[tab][tab]info .describe etc 資料相關資訊
  1. .hist .boxplot etc 
  1. corr相關係數 
  1. value counts 
1. Data Cleaning
     1. missing data count 填補缺失值
1. Data exploration
1. Feature Engineering
1. Data preprocessing for Model
1. Basic Model Building
1. Model Tunging
1. Ensemble Model Building
1. Result
  
## 實作過程觀察到的訊息
首先用 describe 來看資料相關資訊
```python
    print(train.describe())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20describe.PNG?raw=true)

再來用 info 來看資料相關資訊
```python
    print(train.info())
```
![image](https://github.com/winka/IMG/blob/main/tantic%20info.PNG?raw=true)

## 我的觀察

Age, Cabin 欄位有缺失值需要再填補缺失值  

 1. 數值型欄位 .hist   
```python
    for i in train.columns:
        plt.hist(train[i])
        plt.title(i)
        plt.show()
```

| Columns name  | Columns plot  |
| ------------- |:-------------:|
| Sex           | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20sex.png?raw=true)          |
Parch      | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20parch.png?raw=true)             |
| Sibsp           | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20sibsp.png?raw=true)          |
|  Fare     | ![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20fare.png?raw=true)            |
| Age      |![image](https://github.com/winka/IMG/blob/main/tantic%20plot%20age.png?raw=true)         
