import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Users\user\Desktop\학교\3학년\2학기\머신러닝\3주차\PHW-Lab\breast-cancer-wisconsin.csv', index_col=[])

"""
Data preprocessing
"""
col_name = ['sample', 'thickness', 'cell_size',
            'cell_shape', 'adhesion', 'epithelial_cell_size',
            'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'class']
#print(data.info())

# Remove dirty data
data = data.drop(['sample', 'nuclei'], axis=1)
#print(data.info())

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

k = [5, 6, 7, 8, 9]


def BestPerform(X, y, k):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    dtc_gini = DecisionTreeClassifier(criterion='gini')
    dtc_ent = DecisionTreeClassifier(criterion='entropy')
    svm = SVC()
    lr = LogisticRegression(max_iter=10000)

    params_dct = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9]
    }
    params_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100]
    }
    params_svm = {
        'kernel': ['linear', 'poly'],
        'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100]
    }

    kfold = KFold(n_splits=k, shuffle=True)

    grid_gini = GridSearchCV(dtc_gini, param_grid=params_dct, cv=kfold, refit=True)
    grid_gini.fit(X_train, y_train)
    grid_ent = GridSearchCV(dtc_ent, param_grid=params_dct, cv=kfold, refit=True)
    grid_ent.fit(X_train, y_train)
    grid_svm = GridSearchCV(svm, param_grid=params_svm, cv=kfold, refit=True)
    grid_svm.fit(X_train, y_train)
    grid_lr = GridSearchCV(lr, param_grid=params_lr, cv=kfold, refit=True)
    grid_lr.fit(X_train, y_train)

    scores = []

    print("K-Fold: ", k)
    print('==============GridSearchCV(DecisionTreeClassifier(gini))==============')
    print('best parameters : ', grid_gini.best_params_)
    print('best score : ', grid_gini.best_score_)
    em_gini = grid_gini.best_estimator_
    pred_gini = em_gini.predict(X_test)
    scores.append(accuracy_score(y_test, pred_gini))
    print('\n')

    print('==============GridSearchCV(DecisionTreeClassifier(entropy))==============')
    print('best parameters : ', grid_ent.best_params_)
    print('best score : ', grid_ent.best_score_)
    em_ent = grid_ent.best_estimator_
    pred_ent = em_ent.predict(X_test)
    scores.append(accuracy_score(y_test, pred_ent))
    print('\n')

    print('==============GridSearchCV(LogisticRegression)==============')
    print('best parameters : ', grid_lr.best_params_)
    print('best score : ', grid_lr.best_score_)
    em_lr = grid_lr.best_estimator_
    pred_lr = em_lr.predict(X_test)
    scores.append(accuracy_score(y_test, pred_lr))
    print('\n')

    print('==============GridSearchCV(SVM)==============')
    print('best parameters : ', grid_svm.best_params_)
    print('best score : ', grid_svm.best_score_)
    em_svm = grid_svm.best_estimator_
    pred_svm = em_svm.predict(X_test)
    scores.append(accuracy_score(y_test, pred_svm))
    print('\n')

    model_name = [
        'DecisionTreeClassifier(gini)', 'DecisionTreeClassifier(entropy)',
        'LinearRegression', 'SVM'
    ]

    best_index = 0
    for i in range(1, len(scores)):
        if scores[best_index] < scores[i]:
            best_index = i

    print("Model with best performance: ", model_name[best_index], ", ", scores[best_index])
    print("\n\n")


for i in range(len(k)):
    BestPerform(X, y, k[i])


