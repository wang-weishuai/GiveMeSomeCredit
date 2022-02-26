import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



# instream


train = pd.read_csv('./data/what2.csv')
test = pd.read_csv('./test_what2.csv')

train = train.drop('Unnamed: 0', axis=1)
test = test.drop(['Unnamed: 0', 'SeriousDlqin2yrs'], axis=1)


#


def fillMedian(data, data2):
    cols = {}
    for i in data.columns:
        if (data[i].isnull().sum() > 0):
            cols[i] = np.nanmedian(pd.concat([data[i], data2[i]], axis=0))
    for i in cols.keys():
        data[i].fillna(cols[i], inplace=True)
        data2[i].fillna(cols[i], inplace=True)


fillMedian(train, test)


# train models


X = train.drop('SeriousDlqin2yrs', axis=1)
y = train['SeriousDlqin2yrs']


# score


def scoring(clf, X, y):
    fpr, tpr, thresholds = roc_curve(y, clf.predict_proba(X)[:, 1])
    # plt.plot(fpr, tpr, marker = 'o')
    # plt.show()
    roc_auc = auc(fpr, tpr)
    return roc_auc


# RF


clfRand = RandomForestClassifier(max_depth=7, max_features=0.5, criterion='entropy',
                                 n_estimators=160, n_jobs=-1)
clfRand2 = RandomForestClassifier(max_depth=7, max_features=0.5, criterion='entropy',
                                  n_estimators=200, n_jobs=-1)

# b = cross_val_score(cv=5, estimator=clfRand, n_jobs=-1, scoring=scoring, X=X, y=y)

clfRand.fit(X, y)
clfRand2.fit(X, y)

print(scoring(clfRand, X, y))
print(scoring(clfRand2, X, y))

predClfRand = clfRand.predict_proba(test)[:, 1]
predClfRand2 = clfRand2.predict_proba(test)[:, 1]


pred = (predClfRand + predClfRand2) / 2

s = pd.read_csv('./data/ratio_pred.csv')

s['Probability'] = pred
s['Probability'] = [int(x+0.5) for x in s['Probability']]
# s['Score'] = 900 - 600 * s['Probability']
s.to_csv('test_ans.csv', index=False)
