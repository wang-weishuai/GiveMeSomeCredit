

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import re as re
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#数据预处理，降采样
TrainData = pd.read_csv('cs-training.csv')
TestData = pd.read_csv('cs-test.csv')
TrainData['NumberOfDependents'].fillna(TrainData['NumberOfDependents'].median(), inplace=True)
print(TrainData.isnull().sum())

#这一步就是看一下大概的数据分布
train_df.describe()

TrainData.loc[TrainData['age'] < 18]
TrainData.loc[TrainData['age'] == 0, 'age'] = TrainData['age'].median()
TrainData['MonthlyIncome'] = TrainData['MonthlyIncome'].replace(np.nan,TrainData['MonthlyIncome'].mean())
TrainData['NumberOfDependents'].fillna(TrainData['NumberOfDependents'].median(), inplace=True)

#绘制相关系数的热力图
corr = TrainData.corr()
plt.figure(figsize=(19, 15))
sns.heatmap(corr, annot=True, fmt='.2g')

def changedata(column):
    # 改变96与98对应的数据
    new = []
    newval = column.median()
    for i in column:
        if (i != 96 and i != 98):
            new.append(i)
        else:
            new.append(newval)
    return new

# 使用change函数修改数据
TrainData['NumberOfTime30-59DaysPastDueNotWorse'] = changedata(TrainData['NumberOfTime30-59DaysPastDueNotWorse'])
TrainData['NumberOfTimes90DaysLate'] = changedata(TrainData['NumberOfTimes90DaysLate'])
TrainData['NumberOfTime60-89DaysPastDueNotWorse'] = changedata(TrainData['NumberOfTime60-89DaysPastDueNotWorse'])

TestData['NumberOfTime30-59DaysPastDueNotWorse'] = changedata(TestData['NumberOfTime30-59DaysPastDueNotWorse'])
TestData['NumberOfTimes90DaysLate'] = changedata(TestData['NumberOfTimes90DaysLate'])
TestData['NumberOfTime60-89DaysPastDueNotWorse'] = changedata(TestData['NumberOfTime60-89DaysPastDueNotWorse'])

#绘制经过change函数变换后相关系数的热力图

corr = TrainData.corr()
plt.figure(figsize=(19, 15))
sns.heatmap(corr, annot=True, fmt='.2g')

#使用同样的方式填充TestData
TestData.loc[TestData['age'] == 0, 'age'] = TestData['age'].median()
TestData['MonthlyIncome'] = TestData['MonthlyIncome'].replace(np.nan,TestData['MonthlyIncome'].mean())
TestData['NumberOfDependents'].fillna(TestData['NumberOfDependents'].median(), inplace=True)
TrainData.info()


print(train_df.info())

#为了方便后续操作，把原数据集中没有名称的一栏（就是那个1-150000的编号）重命名为ID
TrainData.head(5)
TrainData.rename(columns={'Unnamed: 0':'ID'}, inplace=True)
TestData.rename(columns={'Unnamed: 0':'ID'}, inplace=True)

X = TrainData.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
y = TrainData['SeriousDlqin2yrs']
W = TestData.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
z = TestData['SeriousDlqin2yrs']

X = TrainData.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
y = TrainData['SeriousDlqin2yrs']
W = TestData.drop(['SeriousDlqin2yrs', 'ID'],axis=1)
z = TestData['SeriousDlqin2yrs']

# 划分训练集和测试集
TrainX, TestX, TrainY, TestY = train_test_split(X,y,random_state=111)
scaler = StandardScaler().fit(TrainX)

# 标准化X_train 和X_test
ScaledTrainX = scaler.transform(TrainX)
ScaledTestX = scaler.transform(TestX)


def plot_roc_curve(fpr, tpr, label=None):
    plt.figure(figsize=(12,10))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1], "k--") # 画直线做参考
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    
rus = RandomUnderSampler(random_state=111)

# 直接降采样后返回采样后的数值
# 降采样后违约数据与不违约数据的比例调整成为均衡
X_resampled, y_resampled = rus.fit_sample(X, y)
print ('原始数据集大小 :', Counter(y))
print ('降采样后数据集大小 :', Counter(y_resampled))

X_train_rus, X_test_rus, y_train_rus, y_test_rus = train_test_split(X_resampled, y_resampled, random_state=111)
X_train_rus.shape, y_train_rus.shape

#Logistic模型【听老师的话不用中文了】
logit = LogisticRegression(random_state=111, solver='saga', penalty='l1', class_weight='balanced', C=1.0, max_iter=500)
logit.fit(TrainX, TrainY)

# 输入训练集，返回每个样本对应到每种分类结果的概率
logit_scores_proba = logit.predict_proba(TrainX)

# 返回分类1的概率
logit_scores = logit_scores_proba[:,1]
fpr_logit, tpr_logit, thresh_logit = roc_curve(TrainY, logit_scores)



logit_resampled = LogisticRegression(random_state=111, solver='saga', penalty='l1', class_weight='balanced', C=1.0, max_iter=500)

logit_resampled.fit(X_resampled, y_resampled)
logit_resampled_proba_res = logit_resampled.predict_proba(X_resampled)
logit_resampled_scores = logit_resampled_proba_res[:, 1]
fpr_logit_resampled, tpr_logit_resampled, thresh_logit_resampled = roc_curve(y_resampled, logit_resampled_scores)
plot_roc_curve(fpr_logit_resampled, tpr_logit_resampled)
print ('Logistic模型的AUC score为:', roc_auc_score(y_resampled, logit_resampled_scores))

#RandomForest模型

forest = RandomForestClassifier(n_estimators=300, random_state=111, max_depth=5, class_weight='balanced')
forest.fit(X_train_rus, y_train_rus)
y_scores_prob = forest.predict_proba(X_train_rus)
y_scores = y_scores_prob[:, 1]
fpr, tpr, thresh = roc_curve(y_train_rus, y_scores)
plot_roc_curve(fpr, tpr)
#y_scores.shape
print ('RandomForest模型的AUC score为:', roc_auc_score(y_train_rus, y_scores))

y_test_proba = forest.predict_proba(X_test_rus)
y_scores_test = y_test_proba[:, 1]
fpr_test, tpr_test, thresh_test = roc_curve(y_test_rus, y_scores_test)
plot_roc_curve(fpr_test, tpr_test)
print ('交叉验证后RandomForest模型的AUC score为:', roc_auc_score(y_test_rus, y_scores_test))

###end
