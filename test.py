import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

iris_data = load_iris()
label = iris_data.target
data = iris_data.data
print(type(data))
print(label)
X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2)


# GridSearchCV의 param_grid 설정
params = {
    'max_depth': [2, 3],
    'min_samples_split': [2, 3]
}
paramas_svm = {
    'kernel': ['linear', 'poly'],
    'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100]
}
params_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100]
    }
dtc = DecisionTreeClassifier(criterion='entropy', random_state=None)
svm = SVC()
lr = LogisticRegression(max_iter=10000)

grid_tree = GridSearchCV(dtc, param_grid=params, cv=3, refit=True)
grid_tree.fit(X_train, y_train)

grid_svm = GridSearchCV(svm, param_grid=paramas_svm, cv=5, refit=True)
grid_svm.fit(X_train, y_train)

grid_lr = GridSearchCV(lr, param_grid=params_lr, cv=5, refit=True)
grid_lr.fit(X_train, y_train)


print('best parameters : ', grid_tree.best_params_)
print('best score : ', grid_tree.best_score_)
em = grid_tree.best_estimator_
pred = em.predict(X_val)
accuracy_score(y_val, pred)*100

print('best parameters : ', grid_svm.best_params_)
print('best score : ', grid_svm.best_score_)
em_svm = grid_svm.best_estimator_
pred_svm = em_svm.predict(X_val)
accuracy_score(y_val, pred_svm)

print('best parameters : ', grid_lr.best_params_)
print('best score : ', grid_lr.best_score_)
em_lr = grid_lr.best_estimator_
pred_lr = em_lr.predict(X_val)
accuracy_score(y_val, pred_lr)