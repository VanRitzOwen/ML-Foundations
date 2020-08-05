import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

#data = pd.read_csv('./dataset/winequality-red.csv', sep=';')
data = pd.read_csv('./dataset/winequality-white.csv', sep=';')

print(data.head())
print(data.shape)
print(data.describe())

X = data.drop(['quality'], axis=1)
y = data.quality

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

'''
X_train_scaled = preprocessing.scale(X_train)
print(X_train_scaled.mean(axis=0), X_train_scaled.std(axis=0))
'''

'''
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train_scaled.mean(axis=0), X_train_scaled.std(axis=0))
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled.mean(axis=0), X_test_scaled.std(axis=0))
'''

pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

#joblib.dump(clf, 'model.pkl')