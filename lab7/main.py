import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


class NaiveBayesClassifier:
    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_class = X_train[y_train == c]
            self._mean[idx, :] = X_class.mean(axis=0)
            self._var[idx, :] = X_class.var(axis=0)
            self._priors[idx] = X_class.shape[0] / float(n_samples)

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test.values]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self.probability_density(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def probability_density (self, class_idx, x):
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


def age_to_category(age):
    if age < 18:
        return 0  # child
    elif age < 60:
        return 1  # adult
    else:
        return 2  # senior


def preprocess_data(data):
    if 'Embarked' in data.columns:
        data['Embarked'] = data['Embarked'].apply(embarked_to_num)
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(data['Age'].mean())
        data['Age'] = data['Age'].apply(age_to_category)
    data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    return data

def key_test(features):
    #train

    data = pd.read_csv("train.csv", header=0)
    data = data[features + ['Survived']]
    data = preprocess_data(data)

    X_train = data.drop('Survived', axis='columns')
    y_train = data.Survived

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    #test

    test = pd.read_csv("test.csv", header=0)
    test = test[features]
    test = preprocess_data(test)

    predictions = nb.predict(test)

    test['Survived'] = predictions
    test['PassengerId'] = pd.read_csv("test.csv")['PassengerId']

    test[['PassengerId', 'Survived']].to_csv('result.csv', index=False)

    correct = pd.read_csv("correct.csv")

    accuracy = accuracy_score(correct['Survived'], test['Survived'])
    print(f"Accuracy: {accuracy:.4f}\n")


def cross_val_test(features):
    data = pd.read_csv("train.csv", header=0)
    data = data[features + ['Survived']]
    data = preprocess_data(data)

    X_train = data.drop('Survived', axis='columns')
    y_train = data.Survived

    kf = KFold(n_splits=5, shuffle=True)

    accuracy_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        nb = NaiveBayesClassifier()
        nb.fit(X_train_fold, y_train_fold)

        predictions = nb.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, predictions)
        accuracy_scores.append(accuracy)

    mean_accuracy = np.mean(accuracy_scores)
    print(f"Mean cross-validation accuracy: {mean_accuracy:.4f}")

if __name__ == "__main__":
    features_list = [
        ['Pclass', 'Sex', 'Age', 'Parch', 'Embarked'],
        ['Sex', 'Age', 'Fare', 'SibSp', 'Embarked'],
        ['Sex', 'Age', 'Fare', 'Parch', 'Embarked'],
        ['Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    ]

    for features in features_list:
        print(features)
        cross_val_test(features)
        key_test(features)
