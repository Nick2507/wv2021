import pandas as ps
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

inputFile = "2021_prolog.csv"
colums = ['bmbrPointsLast3Weeks','nmbrPointsFirst3Weeks','binary_score','avDiffRatioBlokFirstMonth','avFirstMonthCompletedSubSubmSolRatio',
                                                       'avBlokSubmSolRatio','avNmbrLong_2','avNmbrQuiteLong_15','avNmbrQuiteShort_075',
                                                       'avNmbrShort_05','averSolSubmImportance3DiffRatio','averSolSubmDiffRatio','averSolSubmDiffRatioAll',
                                                       'averSolSubmDiff','averSolSubmDiffAll','averageEachAssignmentSubmitted','averageEachAssignmentSubmittedSuccess']


def displaysublist(A):
    # store all the sublists
    B = []

    # first loop
    for i in range(len(A) + 1):
        # second loop
        for j in range(i + 1, len(A) + 1):
            # slice the subarray
            sub = A[i:j]
            if (len(sub) > 1 ):
                B.append(sub)
    return B
listperm = displaysublist(colums)
max = ""
maxacc = 0
maxx = []

for x in range(1):
    print(x)

    data = ps.read_csv(inputFile,delimiter=(";"), usecols=['nmbrPointsFirst3Weeks',  'avDiffRatioBlokFirstMonth', 'avFirstMonthCompletedSubSubmSolRatio', 'avBlokSubmSolRatio', 'binary_score'])

    X = data.drop(columns=['binary_score'])
    y = data['binary_score']

    score = 0
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accurance = accuracy_score(y_test,prediction)
        score = score + accurance

    print("De gemiddelde accuraatheid van decision tree")
    score = score/1000
    if score > maxacc:
        maxacc = score
        maxx  =x
        max = "DT"
    print(score)

    score = 0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        log = linear_model.LogisticRegression(max_iter=10000)
        log.fit(X_train,y_train)

        logpred = log.predict(X_test)
        ac = accuracy_score(y_test,logpred)
        score = score + ac
    print('gemiddelde accuraatheid logistische regressie')
    score = score / 100
    print(score)

    score = 0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accurance = accuracy_score(y_test,prediction)
        score = score + accurance

    print("De gemiddelde accuraatheid van random forest")

    score = score / 100
    if score > maxacc:
        maxacc = score
        maxx = x
        max = "RF"
    print(score)


    score = 0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        model = ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=2, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accurance = accuracy_score(y_test,prediction)
        score = score + accurance

    print("De gemiddelde accuraatheid van extra tree")
    score = score / 100
    if score > maxacc:
        maxacc = score
        maxx = x
        max = "ET"
    print(score)

    score = 0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        model = AdaBoostClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accurance = accuracy_score(y_test,prediction)
        score = score + accurance

    print("De gemiddelde accuraatheid van ADA BOOST")
    score = score / 100
    if score > maxacc:
        maxacc = score
        maxx = x
        max = "ADA"
    print(score)

    score = 0
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        accurance = accuracy_score(y_test,prediction)
        score = score + accurance

    print("De gemiddelde accuraatheid van gradient BOOST")
    score = score / 100
    if score > maxacc:
        maxacc = score
        maxx = x
        max = "GB"
    print(score)
print("The Best Model is ")
print(max)
print(maxx)
print(maxacc)