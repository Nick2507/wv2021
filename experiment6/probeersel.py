import pandas as pd
from bioinfokit.analys import get_data, visuz
import seaborn as sns
import numpy as np
import smogn
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_regression

df = pd.read_csv('C:\\Users\\nisse\Desktop\Finale paper WV\\Last3Years\\Prolog_Last3Years.csv')
y = df.iloc[:, 2]
X = df.iloc[:, 3:]


print("X.colum: ", X.columns)
A = X.columns
# make a list of columns: X.columns + ['BinaryScore']
X_new = X.columns.tolist()
X_new.append('TotalScore')
X_new.append('BinaryScore')
X_new = pd.DataFrame(df[X_new])
print(X_new.head(5))

score_smogn = smogn.smoter(
    data=X_new,  ## pandas dataframe
    rel_coef=0.5,  ## relative coef for smote
    samp_method = 'extreme',
    rel_thres = 0.99,
    y='TotalScore'  ## string ('header name')
)

print(score_smogn)
print("X.shape: ", X.shape)
print("score_smogn.shape: ", score_smogn.shape)

# delete the column 'TotalScore' from X
y = score_smogn['BinaryScore']
del score_smogn['TotalScore']
del score_smogn['BinaryScore']
X = score_smogn


print("after deleting the column 'TotalScore'")
print(score_smogn)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_resample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['BinaryScore'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['BinaryScore']==0]))
print("Number of subscription",len(os_data_y[os_data_y['BinaryScore']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['BinaryScore']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['BinaryScore']==1])/len(os_data_X))

data_final_vars = X.columns.values.tolist()
y=['BinaryScore']
X=[i for i in data_final_vars if i not in y]
logreg = LogisticRegression()
rfe = RFE(logreg, step = 200000)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


# new empty set
final_vars =[]
# print every feature and its ranking
for feature, ranking in zip(data_final_vars, rfe.ranking_):
    if ranking == 1:
        final_vars.append(feature)
    print(feature, ranking)
print(final_vars)

X=os_data_X[final_vars]
y=os_data_y['BinaryScore']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))