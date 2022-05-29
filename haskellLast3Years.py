import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

inputFile = "Haskell_Last3YearsDONE.csv"

data = ps.read_csv(inputFile,delimiter=(";"))

X = data.drop(columns=['haskell_score','succes','student_id'])
y = data['succes']

score = 0
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van decision tree")
print(score/1000)

dif = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    log = linear_model.LogisticRegression(max_iter=10000)
    log.fit(X_train,y_train)

    logpred = log.predict(X_test)
    ac = accuracy_score(y_test,logpred)
    dif = dif + ac
print('gemiddelde accuraatheid logistische regressie')
print(dif/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van random forest")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    model = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van extra tree")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van ADA BOOST")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van gradient BOOST")
print(score/100)