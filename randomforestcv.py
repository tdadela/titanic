
import os
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
train_data = pd.read_csv("train.csv")
print(train_data.head())


from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]
test_data = pd.read_csv("test.csv")
test_data['Age'].fillna(test_data['Age'].median(), inplace = True)
train_data['Age'].fillna(train_data['Age'].median(), inplace = True)
features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
# features = ["Sex"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

param_grid = {
    'n_estimators': list(range(50,161, 15)),
    'max_depth': [3,4,5,6,7,9],
}
'''
    'bootstrap': [True],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
'''
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = model, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X, y)
predictions = grid_search.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission3.csv', index=False)
print("Your submission was successfully saved!")
print(f"{grid_search.best_params_=}")
