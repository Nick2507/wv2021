import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from math import sqrt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

inputFile = "Haskell1920_EnkelInformatica.csv"

data = ps.read_csv(inputFile,delimiter=(","))

X = data.drop(columns=['student_id','HaskellScore','BinaryScore'])
y = data['BinaryScore']
print(f"Haskell file is {inputFile}")
print()

vlifOK = ['nb_PointsLast2WeeksHaskell', 'avdiffDeadline.1', 'nmbMorningSubsOnTIme',
'nmbNightSubsOnTIme', 'nmbrCheat.1', 'nmbrCheatOnTime.1',
'nmbMorningubsOnTImeCompleted.1', 'nmbDaySubsOnTImeCompleted.1',
 'avNmbrShort_05', 'avNmbrAroundSol_0.25']
Xvlif = data[vlifOK]

def standaarAfwijking(list):
    n = len(list)
    mean = sum(list)/n
    deviaton = [(x - mean) ** 2 for x in list]
    variance = sum(deviaton)/ float(n)
    standaartAfwijking = sqrt(variance)
    return standaartAfwijking


iteraties = 500

accuraatheid = []
for i in range(iteraties):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
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
