###############################################################################
#                          Stacking Classifier                                #
###############################################################################

# Stacking Classifier Documentation - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
# Mlxtend Documentation - http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier

# Colab Notebook Link - https://colab.research.google.com/drive/1HYZzNPkv1fgJsvOK1fascnxdGzdhbXou?usp=sharing

""" ***************************************************************************
# * File Description:                                                         *
# * An Example of How to use Stack Classifiers.                               *
# *                                                                           *
# * The contents of this script are:                                          *
# * 1. Importing Libraries and Dataset                                        *
# * 2. Feature Engineering and Selection                                      *
# * 3. Data Preprocessing                                                     *
# * 4. Splitting the dataset into training set and test set                   *
# * 5. Random Forest Classifier                                               *
# * 6. Logistic Regression Classifier                                         *
# * 7. Naive Bayes Classifier                                                 *
# * 8. Multi-Layer Perceptron Classifier                                      *
# * 9. Stacking Classifier                                                    *
# * 10. Tuning the Meta-Classifier                                            *
# * --------------------------------------------------------------------------*
# * AUTHOR: Rahul Bordoloi <rahulbordoloi24@gmail.com>                        *
# * --------------------------------------------------------------------------*
# * DATE CREATED: 2nd July, 2020                                              *
# * ************************************************************************"""

###############################################################################
#                    1. Importing Libraries and Dataset                       #
###############################################################################

# Do
'''!pip install category_encoders'''

## For Data Operations and Visualizations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

## For Classifiers
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# Importing Dataset
df = pd.read_csv('Churn_Modelling.csv')

###############################################################################
#                    2. Feature Engineering and Selection                     #
###############################################################################

df.columns

'''
Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
       'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Exited'],
      dtype='object')
'''

# Dropping off redundant columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], inplace = True, axis = 1)  

df.info()

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 11 columns):
CreditScore        10000 non-null int64
Geography          10000 non-null object
Gender             10000 non-null object
Age                10000 non-null int64
Tenure             10000 non-null int64
Balance            10000 non-null float64
NumOfProducts      10000 non-null int64
HasCrCard          10000 non-null int64
IsActiveMember     10000 non-null int64
EstimatedSalary    10000 non-null float64
Exited             10000 non-null int64
dtypes: float64(2), int64(7), object(2)
memory usage: 859.5+ KB
'''
# Check for Imbalance
df.groupby('Exited')['Geography'].count()

'''
Exited
0    7963
1    2037
Name: Geography, dtype: int64
'''

###############################################################################
#                         3. Data Preprocessing                               #
###############################################################################

# Encoding Categorical Variables
l = LabelEncoder()
df['Gender'] = l.fit_transform(df['Gender'])

encoder = TargetEncoder()
df['country'] = encoder.fit_transform(df['Geography'], df['Exited'])

df.drop(['Geography'], inplace = True, axis = 1)

# Spliting into dependent and independent vectors
x = df.drop(['Exited'], axis = 1)
y = df.Exited

# y = y.values.reshape(-1,1)

# Standard Scaling
S = StandardScaler()
x = S.fit_transform(x)

###############################################################################
#         4. Splitting the dataset into training set and test set             #
###############################################################################

x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size = 0.25, 
                                                    random_state = 0)

###############################################################################
#                      5. Random Forest Classifier                            #
###############################################################################

# fitting my model
classifier = rfc(n_estimators = 100, random_state = 0, criterion = 'entropy')
classifier.fit(x_train, y_train)

# predicting the test set results
y_pred = classifier.predict(x_test)

# Checking Accuracy
print(classification_report(y_test, y_pred))

'''
                precision   recall  f1-score   support

           0       0.87      0.96      0.91      1991
           1       0.72      0.45      0.56       509

    accuracy                           0.85      2500
   macro avg       0.80      0.70      0.73      2500
weighted avg       0.84      0.85      0.84      2500
'''

###############################################################################
#                   6. Logistic Regression Classifier                         #
###############################################################################

# fitting my model
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# predicting the test set results
y_pred = classifier.predict(x_test)

# Checking Accuracy
print(classification_report(y_test, y_pred))

'''
                precision   recall  f1-score   support

           0       0.82      0.97      0.89      1991
           1       0.60      0.17      0.27       509

    accuracy                           0.81      2500
   macro avg       0.71      0.57      0.58      2500
weighted avg       0.77      0.81      0.76      2500
'''

###############################################################################
#                        7. Naive Bayes Classifier                            #
###############################################################################

# fitting my model
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# predicting the test set results
y_pred = classifier.predict(x_test)

# Checking Accuracy
print(classification_report(y_test, y_pred))

'''
                precision   recall  f1-score   support

           0       0.83      0.97      0.90      1991
           1       0.70      0.24      0.36       509

    accuracy                           0.82      2500
   macro avg       0.77      0.61      0.63      2500
weighted avg       0.81      0.82      0.79      2500
'''

###############################################################################
#                  8. Multi-Layer Perceptron Classifier                       #
###############################################################################

# fitting my model
classifier = MLPClassifier(activation = "relu", alpha = 0.05, random_state = 0)
classifier.fit(x_train, y_train)

# predicting the test set results
y_pred = classifier.predict(x_test)

# Checking Accuracy
print(classification_report(y_test, y_pred))

'''
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1991
           1       0.75      0.47      0.58       509

    accuracy                           0.86      2500
   macro avg       0.81      0.72      0.75      2500
weighted avg       0.85      0.86      0.85      2500
'''

###############################################################################
#                         9. Stacking Classifier                              #
###############################################################################

# Install Dependencies in py-shell
'''!pip install mlxtend'''

# Importing Necessary Libraries
from sklearn.ensemble import StackingClassifier

# Initialising the Stacking Algorithms
estimators = [
        ('naive-bayes', GaussianNB()),
        ('random-forest', rfc(n_estimators = 100, random_state = 0)),
        ('mlp', MLPClassifier(activation = "relu", alpha = 0.05, random_state = 0))
        ]

# Setting up the Meta-Classifier
clf = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(random_state = 0)
        )
# fitting my model
clf.fit(x_train, y_train)

# getting info about the hyperparameters 
clf.get_params()

'''
{'cv': None,
 'estimators': [('naive-bayes', GaussianNB(priors=None, var_smoothing=1e-09)),
  ('random-forest',
   RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                          criterion='gini', max_depth=None, max_features='auto',
                          max_leaf_nodes=None, max_samples=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=0, verbose=0,
                          warm_start=False)),
  ('mlp',
   MLPClassifier(activation='relu', alpha=0.05, batch_size='auto', beta_1=0.9,
                 beta_2=0.999, early_stopping=False, epsilon=1e-08,
                 hidden_layer_sizes=(100,), learning_rate='constant',
                 learning_rate_init=0.001, max_fun=15000, max_iter=200,
                 momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                 power_t=0.5, random_state=0, shuffle=True, solver='adam',
                 tol=0.0001, validation_fraction=0.1, verbose=False,
                 warm_start=False))],
 'final_estimator': LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False),
 'final_estimator__C': 1.0,
 'final_estimator__class_weight': None,
 'final_estimator__dual': False,
 'final_estimator__fit_intercept': True,
 'final_estimator__intercept_scaling': 1,
 'final_estimator__l1_ratio': None,
 'final_estimator__max_iter': 100,
 'final_estimator__multi_class': 'auto',
 'final_estimator__n_jobs': None,
 'final_estimator__penalty': 'l2',
 'final_estimator__random_state': 0,
 'final_estimator__solver': 'lbfgs',
 'final_estimator__tol': 0.0001,
 'final_estimator__verbose': 0,
 'final_estimator__warm_start': False,
 'mlp': MLPClassifier(activation='relu', alpha=0.05, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(100,), learning_rate='constant',
               learning_rate_init=0.001, max_fun=15000, max_iter=200,
               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
               power_t=0.5, random_state=0, shuffle=True, solver='adam',
               tol=0.0001, validation_fraction=0.1, verbose=False,
               warm_start=False),
 'mlp__activation': 'relu',
 'mlp__alpha': 0.05,
 'mlp__batch_size': 'auto',
 'mlp__beta_1': 0.9,
 'mlp__beta_2': 0.999,
 'mlp__early_stopping': False,
 'mlp__epsilon': 1e-08,
 'mlp__hidden_layer_sizes': (100,),
 'mlp__learning_rate': 'constant',
 'mlp__learning_rate_init': 0.001,
 'mlp__max_fun': 15000,
 'mlp__max_iter': 200,
 'mlp__momentum': 0.9,
 'mlp__n_iter_no_change': 10,
 'mlp__nesterovs_momentum': True,
 'mlp__power_t': 0.5,
 'mlp__random_state': 0,
 'mlp__shuffle': True,
 'mlp__solver': 'adam',
 'mlp__tol': 0.0001,
 'mlp__validation_fraction': 0.1,
 'mlp__verbose': False,
 'mlp__warm_start': False,
 'n_jobs': None,
 'naive-bayes': GaussianNB(priors=None, var_smoothing=1e-09),
 'naive-bayes__priors': None,
 'naive-bayes__var_smoothing': 1e-09,
 'passthrough': False,
 'random-forest': RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                        criterion='gini', max_depth=None, max_features='auto',
                        max_leaf_nodes=None, max_samples=None,
                        min_impurity_decrease=0.0, min_impurity_split=None,
                        min_samples_leaf=1, min_samples_split=2,
                        min_weight_fraction_leaf=0.0, n_estimators=100,
                        n_jobs=None, oob_score=False, random_state=0, verbose=0,
                        warm_start=False),
 'random-forest__bootstrap': True,
 'random-forest__ccp_alpha': 0.0,
 'random-forest__class_weight': None,
 'random-forest__criterion': 'gini',
 'random-forest__max_depth': None,
 'random-forest__max_features': 'auto',
 'random-forest__max_leaf_nodes': None,
 'random-forest__max_samples': None,
 'random-forest__min_impurity_decrease': 0.0,
 'random-forest__min_impurity_split': None,
 'random-forest__min_samples_leaf': 1,
 'random-forest__min_samples_split': 2,
 'random-forest__min_weight_fraction_leaf': 0.0,
 'random-forest__n_estimators': 100,
 'random-forest__n_jobs': None,
 'random-forest__oob_score': False,
 'random-forest__random_state': 0,
 'random-forest__verbose': 0,
 'random-forest__warm_start': False,
 'stack_method': 'auto',
 'verbose': 0}
'''

# predicting the test set results
y_pred = clf.predict(x_test)

# Checking Accuracy
print(classification_report(y_test, y_pred))

'''
                precision   recall  f1-score   support

           0       0.88      0.96      0.92      1991
           1       0.76      0.46      0.58       509

    accuracy                           0.86      2500
   macro avg       0.82      0.71      0.75      2500
weighted avg       0.85      0.86      0.85      2500
'''

###############################################################################
#                     10. Tuning the Meta-Classifier                          #
###############################################################################

# Defining Parameter Grid
params = {'final_estimator__C': [1.0,1.1,1.5],
          'final_estimator__max_iter': [50,100,150,200],
          'final_estimator__n_jobs': [1,-1,5],
          'final_estimator__penalty': ['l1','l2'],
          'final_estimator__random_state': [0],
          }

# Initialize GridSearchCV
grid = GridSearchCV(estimator = clf, 
                    param_grid = params, 
                    cv = 5,
                    scoring = "roc_auc",
                    verbose = 10,
                    n_jobs = -1)

# Fit GridSearchCV
grid.fit(x_train, y_train)

'''
Fitting 5 folds for each of 72 candidates, totalling 360 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=-1)]: Done   1 tasks      | elapsed:   48.6s
[Parallel(n_jobs=-1)]: Done   4 tasks      | elapsed:  1.6min
[Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:  4.0min
[Parallel(n_jobs=-1)]: Done  14 tasks      | elapsed:  5.5min
[Parallel(n_jobs=-1)]: Done  21 tasks      | elapsed:  8.7min
[Parallel(n_jobs=-1)]: Done  28 tasks      | elapsed: 11.0min
[Parallel(n_jobs=-1)]: Done  37 tasks      | elapsed: 15.0min
[Parallel(n_jobs=-1)]: Done  46 tasks      | elapsed: 18.2min
[Parallel(n_jobs=-1)]: Done  57 tasks      | elapsed: 22.9min
[Parallel(n_jobs=-1)]: Done  68 tasks      | elapsed: 27.0min
[Parallel(n_jobs=-1)]: Done  81 tasks      | elapsed: 32.5min
[Parallel(n_jobs=-1)]: Done  94 tasks      | elapsed: 37.3min
[Parallel(n_jobs=-1)]: Done 109 tasks      | elapsed: 43.5min
[Parallel(n_jobs=-1)]: Done 124 tasks      | elapsed: 49.2min
[Parallel(n_jobs=-1)]: Done 141 tasks      | elapsed: 56.2min
[Parallel(n_jobs=-1)]: Done 158 tasks      | elapsed: 62.7min
[Parallel(n_jobs=-1)]: Done 177 tasks      | elapsed: 70.6min
[Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed: 77.7min
[Parallel(n_jobs=-1)]: Done 217 tasks      | elapsed: 86.4min
[Parallel(n_jobs=-1)]: Done 238 tasks      | elapsed: 94.4min
[Parallel(n_jobs=-1)]: Done 261 tasks      | elapsed: 103.7min
[Parallel(n_jobs=-1)]: Done 284 tasks      | elapsed: 112.5min
[Parallel(n_jobs=-1)]: Done 309 tasks      | elapsed: 122.6min
[Parallel(n_jobs=-1)]: Done 334 tasks      | elapsed: 132.1min
[Parallel(n_jobs=-1)]: Done 360 out of 360 | elapsed: 142.3min finished
GridSearchCV(cv=5, error_score=nan,
             estimator=StackingClassifier(cv=None,
                                          estimators=[('naive-bayes',
                                                       GaussianNB(priors=None,
                                                                  var_smoothing=1e-09)),
                                                      ('random-forest',
                                                       RandomForestClassifier(bootstrap=True,
                                                                              ccp_alpha=0.0,
                                                                              class_weight=None,
                                                                              criterion='gini',
                                                                              max_depth=None,
                                                                              max_features='auto',
                                                                              max_leaf_nodes=None,
                                                                              max_samples=None,
                                                                              min_impurity_decrease=0.0,
                                                                              min_i...
                                          stack_method='auto', verbose=0),
             iid='deprecated', n_jobs=-1,
             param_grid={'final_estimator__C': [1.0, 1.1, 1.5],
                         'final_estimator__max_iter': [50, 100, 150, 200],
                         'final_estimator__n_jobs': [1, -1, 5],
                         'final_estimator__penalty': ['l1', 'l2'],
                         'final_estimator__random_state': [0]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='roc_auc', verbose=10)
'''

# predicting the test set results
y_pred = grid.predict(x_test)

# Checking Accuracy
print(classification_report(y_test, y_pred))

'''
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1991
           1       0.75      0.46      0.57       509

    accuracy                           0.86      2500
   macro avg       0.81      0.71      0.75      2500
weighted avg       0.85      0.86      0.85      2500
'''

###############################################################################
#                                 END                                         #
###############################################################################

