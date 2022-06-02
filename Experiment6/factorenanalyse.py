import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from math import sqrt
import numpy as np
import smogn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier



'''
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
'''
'''
vlifOK = ['av_CompletedPrevYears', 'av_StyleLengthCompletedImportance2', 'av_CompletedImportance2SubsDiffRatio',
'nb_Importance2SuccessTests', 'av_StyleLengthCompletedImportance3', "nb_completed'Extra'Subs", 'nb_PointsLast2WeeksHaskell',  \
'nmbrCompleteDecemberSubs', 'nmbrSuccessTestsJanuary', 'nmbNightSubsOnTIme' ,'nmbrCheatOnTime.1', 'nmbMorningubsOnTImeCompleted.1',
'nbMorningCompleteSubs',  'nbNightCompleteSubs', 'nmbrOctoberSubs', 'avNmbrShort_05', \
'avNmbrQuiteShort_075', 'avNmbrQuiteLong_15', 'avNmbrLong_2', 'avNmbrAroundSol_0.25',  'avStyleLengthCompltedSubsJanuary',\
'avStyleLengthCompltedSubsOctober']
'''
vlifOK = ['av_CompletedImportance2SubsDiffRatio', 'nb_Importance2SuccessTests',
"nb_completed'Extra'Subs", 'nb_PointsLast2WeeksHaskell', 'nmbNightSubsOnTIme',
 'nmbrCheatOnTime.1', 'nmbMorningubsOnTImeCompleted.1', 'nbMorningCompleteSubs',
'nbNightCompleteSubs', 'nmbrOctoberSubs', 'avStyleLengthCompltedSubsJanuary']
inputFile = 'C:\\Users\\nisse\Desktop\Finale paper WV\\2021PerLingo\\2021_haskell.csv'
data = pd.read_csv(inputFile,delimiter=(","))
#X = data.drop(columns=['student_id','TotalScore','BinaryScore'])
#y = data['BinaryScore']
Xvlif = data[vlifOK]
X = data;
# drop the columns in X that are not in vlifOK, except for student_id and TotalScore and BinaryScore
for column in X.columns:
    if (not column == 'TotalScore' and not column == 'BinaryScore') and (column not in vlifOK):
            X = X.drop(columns=column)

y = data['BinaryScore']
print(f"Haskell 2021 file is {inputFile}")
print()

Xvlif = data[vlifOK]

def standaarAfwijking(list):
    n = len(list)
    mean = sum(list)/n
    deviaton = [(x - mean) ** 2 for x in list]
    variance = sum(deviaton)/ float(n)
    standaartAfwijking = sqrt(variance)
    return standaartAfwijking
def write(name, listOfAccuracies):
    with open(name, 'w') as f:
        for item in listOfAccuracies:
            f.write("%s\n" % item)
        f.close()

iteraties = 100

data_lijst = []
accuraatheid = []
for i in range(iteraties):
    score_smogn = smogn.smoter(
        data=X,  ## pandas dataframe
        rel_coef=0.5,  ## relative coef for smote
        samp_method='extreme',
        rel_thres=0.99,
        y='TotalScore'  ## string ('header name')
    )
    y = score_smogn['BinaryScore']
    del score_smogn['TotalScore']
    del score_smogn['BinaryScore']
    Xvlif = score_smogn
    os = SMOTE(random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(Xvlif, y, test_size=0.25)
    columns = X_train.columns
    os_data_X, os_data_y = os.fit_resample(X_train, y_train)
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=['BinaryScore'])
    print("length of oversampled data is ", len(os_data_X))
    print("Number of no subscription in oversampled data", len(os_data_y[os_data_y['BinaryScore'] == 0]))
    print("Number of subscription", len(os_data_y[os_data_y['BinaryScore'] == 1]))
    print("Proportion of no subscription data in oversampled data is ",
          len(os_data_y[os_data_y['BinaryScore'] == 0]) / len(os_data_X))
    print("Proportion of subscription data in oversampled data is ",
          len(os_data_y[os_data_y['BinaryScore'] == 1]) / len(os_data_X))

    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    data_lijst.append([X_train, y_train, X_test, y_test, model])
    prediction = model.predict(X_test)
    accurance = accuracy_score(y_test,prediction)
    accuraatheid.append(accurance)
print()
print("Logistische regresie vlif oke")
write("LogistischeRegresieVlifOK.txt", accuraatheid)
gemiddeldeAcc = (sum(accuraatheid)/(len(accuraatheid)))
print(f"De gemiddelde accuraatheid is {gemiddeldeAcc}")
print(f'De maximale accuraatheid {max(accuraatheid)}')
print(f'De minimale accuraatheid {min(accuraatheid)}')
print(f"De standaardafweiking {standaarAfwijking(accuraatheid)}")

# find the index of the median accuracy in accuraatheid
accuraatheid.sort()
index = int(len(accuraatheid)/2)
print(f"De mediane accuraatheid is {accuraatheid[index]}")
y = data_lijst[index][1]
x = data_lijst[index][0]
Y_test = data_lijst[index][3]
X_test = data_lijst[index][2]
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary2())

print(result.summary2().as_latex())
# print the intercept and the coefficients
print()
print("probabilities for SPSS")
proba = 1 / (1 + np.exp( - result.fittedvalues ))
#write proba to file
write("proba.txt", proba)
