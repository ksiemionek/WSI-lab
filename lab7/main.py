import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X.values]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        x = np.array(x, dtype=np.float64)
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp((-(x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


def embarked_to_num(embarked):
    if embarked == "C":
        return 0
    elif embarked == "Q":
        return 1
    else:
        return 2


def key_test(features):
    data = pd.read_csv("train.csv", header=0)
    data = data[features + ['Survived']]

    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].apply(embarked_to_num)

    X_train = data.drop('Survived', axis='columns')
    y_train = data.Survived

    X_train = pd.concat([X_train, pd.get_dummies(X_train.Sex)], axis='columns')
    X_train.drop(['Sex', 'female'], axis='columns', inplace=True)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    test = pd.read_csv("test.csv", header=0)
    test = test[features]

    if 'Embarked' in test.columns:
        test['Embarked'] = test['Embarked'].apply(embarked_to_num)

    test = pd.concat([test, pd.get_dummies(test.Sex)], axis='columns')
    test.drop(['Sex', 'female'], axis='columns', inplace=True)

    if 'Age' in test.columns:
        test['Age'] = test['Age'].fillna(X_train['Age'].mean())

    predictions = nb.predict(test)

    test['Survived'] = predictions
    test['PassengerId'] = pd.read_csv("test.csv")['PassengerId']

    test[['PassengerId', 'Survived']].to_csv('result.csv', index=False)

    correct = pd.read_csv("correct.csv")

    accuracy = accuracy_score(correct['Survived'], test['Survived'])
    print(f"Dokładność modelu: {accuracy:.4f}")


def cross_val_test(features):
    data = pd.read_csv("train.csv", header=0)
    data = data[features + ['Survived']]

    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].apply(embarked_to_num)

    X_train = data.drop('Survived', axis='columns')
    y_train = data.Survived

    X_train['Sex'] = X_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    kf = KFold(n_splits=5, shuffle=True)

    accuracy_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        nb = NaiveBayes()
        nb.fit(X_train_fold, y_train_fold)

        predictions = nb.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, predictions)
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    print(f"Średnia dokładność w ramach walidacji krzyżowej: {mean_accuracy:.4f}")

if __name__ == "__main__":
    # features = ['Sex', 'Fare', 'SibSp', 'Parch', 'Embarked']
    # features = ['Pclass', 'Sex', 'Fare', 'SibSp', 'Embarked']
    # features = ['Pclass', 'Sex', 'Fare', 'SibSp', 'Parch']
    # features = ['Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

    features_list = [
        ['Sex', 'Fare', 'SibSp', 'Parch', 'Embarked'],
        ['Pclass', 'Sex', 'Fare', 'SibSp', 'Embarked'],
        ['Pclass', 'Sex', 'Fare', 'SibSp', 'Parch'],
        ['Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    ]

    for features in features_list:
        key_test(features)
        cross_val_test(features)