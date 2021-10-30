1. Project plan
1) Preprocessing
# Drop unnecessary column
   Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
10. Mitoses                       1 - 10
11. Class:                        (2 for benign, 4 for malignant)
=> We don’t need first attribute, Sample code number. so, it will be dropped.

# Data scaling
It will be used…
- Standard scaling
- Robust scaling
- MinMax Scaling
- MaxAbs Scaling

#Data Encoding
It will be used…
- OneHot Encoding


2) Data analysis,
# Model building
It will be used…
- Decision Tree Classifier (Entropy / Gini)
- Logistic Regression
- Support Vector Machine
with various parameters and hyperparameters.

3) Evaluation
# Testing
- K-Fold Cross Validation with various K


2. Program structure
# Program Structure
Call CompPerform()
CompPerform()
Scale X for in scalers [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
Encode y for in OneHotEncoder()
Modeling data for in models [DecisionTreeClassifier(), LogisticRegression(), svm.SVC()]
Evaluate with KFold(), cross_val_score()

# Main function
CompPerform(X, y, encode_col, scaled_col)
Compare performances of scaler, encoder, fitting algorithm.
Print each accuracies and best one.
- Scalers: [StandardScaler(), RobustScaler(), MinMaxScaler(), MaxAbsScaler()]
- Encoder: OneHotEncoder()
- Models: [DecisionTreeClassifier(), LogisticRegression(), svm.SVC()]
- Evaluate: KFold(), cross_val_score()
Parameters==========
X: Dataframe to be scaled
y: Dataframe to be encoded
encode_col: columns to encode
scaled_col: columns to scaled



