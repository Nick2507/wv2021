import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from math import sqrt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

inputFile = "2021_haskell.csv"

data = ps.read_csv(inputFile,delimiter=(","))

X = data.drop(columns=['student_id','TotalScore','BinaryScore'])
y = data['BinaryScore']
print(f"Haskell 2021 file is {inputFile}")
print()

vlifOK = ['Tabel (5).nmbrImportance2Ex', 'Tabel (7).nmbrPointsAwardedImportance3',
 'Tabel (8).nmbrPointsAwardedImportance2', 'Tabel (11).nmbrSubmissionsLastWeek',
 'Tabel (17).averSolSubmDiff', 'Tabel (23).nmbrSucessTestImportance0', 'Tabel (27).nmbrPointsLast3Weeks']
Xvlif = data[vlifOK]

def standaarAfwijking(list):
    n = len(list)
    mean = sum(list)/n
    deviaton = [(x - mean) ** 2 for x in list]
    variance = sum(deviaton)/ float(n)
    standaartAfwijking = sqrt(variance)
    return standaartAfwijking


iteraties = 100

accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)

print("Decision tree")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")



accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Logistische regresie")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")


accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.25)
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Logistische regresie vlif oke")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")


accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Random Forest")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")


accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.25)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Random forest vlif oke")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")

accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Extra tree")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")


accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.25)
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Extra tree vlif oke")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")

accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("adaBoost")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")


accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.25)
    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("adaBoost vlif oke")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")

accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("GradientBoosting")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")


accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.25)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Gradient boosting vlif oke")
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")