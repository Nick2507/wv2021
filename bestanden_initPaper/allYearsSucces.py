import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

input ="AllYearsWithSucces.csv"
data = ps.read_csv(input,delimiter=(";"))
haskelX = data.drop(columns=['haskell_score','succesH','succesP','prolog_score','student_id','totaal','succes'])
haskelY = data['succesH']

prologX = data.drop(columns=['haskell_score','succesH','succesP','prolog_score','student_id','totaal','succes'])
prologY = data['succesP']

algemeenX = data.drop(columns=['haskell_score','succesH','succesP','prolog_score','student_id','totaal','succes'])
algemeenY = data['succes']

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(haskelX, haskelY, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van decision tree voor haskell")
print(score/100)
score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(prologX, prologY, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van decision tree voor prolog")
print(score/100)
score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(algemeenX, algemeenY, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van decision tree ")
print(score/100)

dif = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(haskelX, haskelY, test_size=0.2)
    log = linear_model.LogisticRegression(max_iter=10000)
    log.fit(X_train,y_train)

    logpred = log.predict(X_test)
    ac = accuracy_score(y_test,logpred)
    dif = dif + ac
print('gemiddelde accuraatheid logistische regressie voor haskel')
print(dif/100)

dif = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(prologX, prologY, test_size=0.2)
    log = linear_model.LogisticRegression(max_iter=10000)
    log.fit(X_train,y_train)

    logpred = log.predict(X_test)
    ac = accuracy_score(y_test,logpred)
    dif = dif + ac
print('gemiddelde accuraatheid logistische regressie voor prolog')
print(dif/100)

dif = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(algemeenX, algemeenY, test_size=0.2)
    log = linear_model.LogisticRegression(max_iter=10000)
    log.fit(X_train,y_train)

    logpred = log.predict(X_test)
    ac = accuracy_score(y_test,logpred)
    dif = dif + ac
print('gemiddelde accuraatheid logistische regressie')
print(dif/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(haskelX, haskelY, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van random forest voor Haskell")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(prologX, prologY, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van random forest voor prolog")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(algemeenX, algemeenY, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van random forest")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(haskelX, haskelY, test_size=0.2)
    model = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van extra tree voor haskell")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(prologX, prologY, test_size=0.2)
    model = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van extra tree voor prolog")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(algemeenX, algemeenY, test_size=0.2)
    model = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van extra tree")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(haskelX, haskelY, test_size=0.2)
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van ADA BOOST haskell")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(prologX, prologY, test_size=0.2)
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van ADA BOOST prolog")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(algemeenX, algemeenY, test_size=0.2)
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van ADA BOOST ")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(haskelX, haskelY, test_size=0.2)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van gradient BOOST haskell")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(prologX, prologY, test_size=0.2)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van gradient BOOST prolog")
print(score/100)

score = 0
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(algemeenX, algemeenY, test_size=0.2)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    score = score + accurance

print("De gemiddelde accuraatheid van gradient BOOST prolog")
print(score/100)