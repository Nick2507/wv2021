import itertools
from random import randint

import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler


inputFile = "Prolog_Last3YearsDONE.csv"

data = ps.read_csv(inputFile,delimiter=(";"))

X = data.drop(columns=['prolog_score','binary_score','student_id'])
k = ['prolog_score','binary_score','student_id']
original = k.copy()
kolomen = list(X.columns)
y = data['binary_score']
best = kolomen.copy()
best_acc = 0
bpop=[]
first = kolomen.copy()
f = True
old_kolomen = kolomen.copy()
old_acc = 0
succes = True
while succes:
    l = len(first)
    for z in range(l):
        print(z)
        cf = first.copy()
        pop = cf.pop(z)
        bpop.append(pop)
        for x in bpop:
            k.append(x)
        popdata = data.drop(columns=k)
        score = 0
        for i in range(50):
            X_train, X_test, y_train, y_test = train_test_split(popdata, y, test_size=0.25)
            pipe = make_pipeline(StandardScaler(),DecisionTreeClassifier())
            pipe.fit(X_train, y_train)
            prediction = pipe.score(X_test,y_test)
            score = score + prediction
        nieuw_acc  = score/50
        if (nieuw_acc > old_acc):
            bestp = pop
            old_kolomen = cf.copy()
            old_acc = nieuw_acc
        else:
            kolomen = old_kolomen.copy()
        k = original.copy()

    if (old_acc > best_acc):
        best = old_kolomen.copy()
        first = old_kolomen.copy()
        best_acc = old_acc
        bpop.append(bestp)
        f = True
    elif f:
        f = False
        first = old_kolomen.copy()
    else:
        succes = False



    if (len(kolomen)==2):
        succes = False

print("Best")
print(best)
print(len(best))
print(best_acc)
