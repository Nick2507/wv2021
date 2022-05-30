import itertools
from random import randint
import copy
import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import StandardScaler

inputFile = "2021_haskell.csv"

data = ps.read_csv(inputFile, delimiter=(","))

X = data.drop(columns=['student_id','TotalScore','BinaryScore'])
k = ['student_id','TotalScore','BinaryScore']
modelen = [DecisionTreeClassifier(),linear_model.LogisticRegression(), RandomForestClassifier(),ExtraTreesClassifier(),AdaBoostClassifier(),GradientBoostingClassifier()]
for x in modelen:
    print(str(x))
    original = k.copy()
    kolomen = list(X.columns)
    y = data['BinaryScore']
    best_acc = 0
    best = []
    succes = True
    semi = 0
    s = []
    semi_elements = []
    semiel = ""

    while succes:
        bestpop = None
        semipop = None
        for i in range(len(kolomen)):
            popelem = (kolomen.copy()).pop(i)
            ck = k.copy()
            ck.append(popelem)
            XK = data.drop(columns=ck)
            score = 0
            for i in range(100):
                X_train, X_test, y_train, y_test = train_test_split(XK, y, test_size=0.25)
                pipe = make_pipeline(StandardScaler(),x)
                pipe.fit(X_train, y_train)
                prediction = pipe.score(X_test, y_test)
                score = score + prediction
            acc = score / 100
            if (acc > best_acc):
                best_acc = copy.copy(acc)
                semi = copy.copy(semi)
                best = kolomen.copy()
                best.remove(popelem)
                s = best.copy()
                bestpop = copy.copy(popelem)
            elif(acc > best_acc - 0.05 and acc > semi ):
                semi = copy.copy(acc)
                s = kolomen.copy()
                s.remove(popelem)
                semipop = copy.copy(popelem)

        if (bestpop != None):
            k.append(bestpop)
            kolomen.remove(bestpop)
            bestpop = None
        elif(semipop != None):
            k.append(semipop)
            kolomen.remove(semipop)
            semiel = "semi: " + semipop
            semi_elements.append(semiel)
            semi = 0

        else:
            succes = False
    print("Best")
    print(best_acc)
    print(best)
    print(len(best))
    print(k)
    print(semi_elements)
    print(semi)
