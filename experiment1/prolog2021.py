import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from math import sqrt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

inputFile = "2021_prolog.csv"

data = ps.read_csv(inputFile,delimiter=(","))

X = data.drop(columns=['student_id','TotalScore','BinaryScore'])
y = data['BinaryScore']
print(f"Prolog 2021 file is {inputFile}")
print()

vlifOK = ['av_ErrorLength', 'av_MockSubs',
          'nb_CompleteImportance3Subs',
          'av_CompleteImportance3DiffRatio',
          'nb_JanuaryImportance3SuccessTests',
            'nmbrCompleteDecemberSubs',
                  'nmbrCheat.1',
         'nmbrCheatOnTime.1', 'nmbMorningubsOnTImeCompleted',
          'nmbNightSubsOnTImeCompleted',
         'nbMorningCompleteSubs',
          'nmbNightSubs', 'nmbrOctoberSubs', 'avNmbrShort_05','avNmbrQuiteShort_075','avNmbrQuiteLong_15',
          'avNmbrLong_2', 'avNmbrAroundSol_0.25', 'avStyleLengthCompltedSubsJanuary', 'avStyleLengthCompltedSubsOctober']
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