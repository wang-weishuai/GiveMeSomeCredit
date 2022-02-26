import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy import stats
import copy

train_data = pd.read_csv('cs-test.csv')
train_data = train_data.iloc[:, 1:]
train_data['NumberOfDependents'].fillna(train_data['NumberOfDependents'].median(), inplace=True)

mData = train_data.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
train_known = mData[mData.MonthlyIncome.notnull()].iloc[:,:].values
train_unknown = mData[mData.MonthlyIncome.isnull()].iloc[:,:].values

train_X = train_known[:, 1:]
train_y = train_known[:, 0]
rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
rfr.fit(train_X, train_y)
predicted_y = rfr.predict(train_unknown[:, 1:]).round(0)
train_data.loc[train_data.MonthlyIncome.isnull(), 'MonthlyIncome'] = predicted_y

train_data = train_data.dropna()


print(train_data.info())
train_data.to_csv('test1.csv')