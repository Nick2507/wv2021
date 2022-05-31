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

controle = ['Tabel.nmbrExtraEx', 'Tabel.nmbrExtraTestSuccess', 'Tabel (2).nmbrExtraExPoints', 'Tabel (2).nmbrExtraTestSuccessPoints', 'Tabel (3).nmbrImportance3Ex', 'Tabel (4).nmbrImportance3ExPoints', 'Tabel (4).nmbrImportance3ExSuccessPoints', 'Tabel (5).nmbrImportance2ExSuccess', 'Tabel (6).nmbrImportance2ExPoints', 'Tabel (6).nmbrImportance2ExPointsSuccessPoints', 'Tabel (7).nmbrPointsAwardedImportance3', 'Tabel (8).nmbrPointsAwardedImportance2', 'Tabel (9).nmbrPointsAwardedOnExtraEx', 'Tabel (10).averageEachAssignmentSubmitted', 'Tabel (11).nmbrSubmissionsLastWeek', 'Tabel (12).nmbrExtraExCompleted', 'Tabel (13).nmbrImportance2ExCompleted', 'Tabel (15).nmbrImportance0ExCompleted', 'Tabel (17).averSolSubmDiff', 'Tabel (18).nmbrSuccessSubmissionsLastWeek', 'Tabel (22).nmbrSuccesTestsLastWeek', 'Tabel (23).nmbrSucessTestImportance0', 'Tabel (24).nmbrSucesTestsImportance2', 'Tabel (25).nmbrSuccesTestsImportance3', 'Tabel (26).nmbrSubmissionsLastWeek', 'Tabel (27).nmbrPointsLast3Weeks', 'Tabel (28).nmbrPointsFirst3Weeks', 'Tabel (29).nmbrPoints', 'Tabel (30).averSolSubmDiffRatio', 'Tabel (31).averSolSubmImportance3DiffRatio', 'Tabel (32).useraverSolSubmImportance2DiffRatio', 'Tabel (33).useruseraverSolSubmImportance0DiffRatio', 'Tabel (34).nmbrCheat', 'Tabel (35).avTimeBeforeExam']

Xcontrole = data[controle]

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
    X_train, X_test, y_train, y_test = train_test_split(Xcontrole, y, test_size=0.25)
    model = AdaBoostClassifier()
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