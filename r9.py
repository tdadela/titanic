
'''
import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        '''

from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
train_data = pd.read_csv("train.csv")
# print(train_data.head())
train_data.groupby(['Pclass', 'Sex'])['Age'].median()


y = train_data["Survived"]
test_data = pd.read_csv("test.csv")
train_data['Age'] = train_data.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.replace(np.nan, x.median()))
test_data['Age'] = test_data.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.replace(np.nan, x.median()))
train_data['Cabin'].fillna('U', inplace=True)
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])
test_data['Cabin'].fillna('U', inplace=True)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])


replacement = {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}
for data in [test_data, train_data]:
    data['Cabin'] = data['Cabin'].apply(lambda x: replacement.get(x))
    data['Cabin'] = StandardScaler().fit_transform(
        data['Cabin'].values.reshape(-1, 1))
#train_data.groupby(['Pclass', 'Sex'])['Age'].median()
#test_data.groupby(['Pclass', 'Sex'])['Age'].median()
#test_data['Age'].fillna(test_data.groupby(['Pclass', 'Sex'])['Age'].median(), inplace = True)
#train_data['Age'].fillna(train_data.groupby(['Pclass', 'Sex'])['Age'].median(), inplace = True)
#test_data['Age'].fillna(test_data['Age'].median(), inplace = True)
#train_data['Age'].fillna(train_data['Age'].median(), inplace = True)
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Cabin"]
# features = ["Sex"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

print(X, X_test)
param_grid = {
    'n_estimators': list(range(20, 101, 15)),
    'max_depth': [3, 4, 5, 6, 7, 9],
}
'''
    'bootstrap': [True],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
'''
model = RandomForestClassifier()  # n_estimators=100, max_depth=5, random_state=1)
#from xgboost import XGBClassifier
#model = XGBClassifier()


#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()#max_iter=3000)

grid_search = model  # GridSearchCV(estimator = model, param_grid = param_grid)
#                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X, y)

#model.fit(X, y)
# print("xxx", X[0], X_test[0])
predictions = grid_search.predict(X_test)

output = pd.DataFrame(
    {'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission22.csv', index=False)
print("Your submission was successfully saved!")
# print(f"{grid_search.best_params_=}")
