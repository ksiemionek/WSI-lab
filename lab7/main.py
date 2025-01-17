import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from data import train_data, test_data


class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X_train, y_train):
        self.class_probs = dict(y_train.value_counts(normalize=True))

        for feature in X_train:
            self.feature_probs[feature] = {}
            for class_val in self.class_probs.keys():
                X_class = X_train[y_train == class_val]
                self.feature_probs[feature][class_val] = dict(
                    X_class[feature].value_counts(normalize=True)
                )

    def predict(self, X_test):
        predictions = []

        for _, row in X_test.iterrows():
            class_scores = {}
            for class_val in self.class_probs.keys():
                score = self.class_probs[class_val]
                for feature in X_test:
                    prob = self.feature_probs[feature][class_val].get(
                        row[feature], 0.001
                    )
                    score *= prob
                class_scores[class_val] = score

            total_score = sum(class_scores.values())
            probs = {
                class_val: class_scores[class_val] / total_score
                for class_val, score in class_scores.items()
            }
            predictions.append(max(probs, key=probs.get))
        return np.array(predictions)


def key_test(features):
    #train

    X_train, y_train = train_data(features)

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    #test

    test = test_data(features)
    predictions = nb.predict(test)

    test['Survived'] = predictions
    test['PassengerId'] = pd.read_csv("test.csv")['PassengerId']

    test[['PassengerId', 'Survived']].to_csv('result.csv', index=False)

    correct = pd.read_csv("correct.csv")

    accuracy = accuracy_score(correct['Survived'], test['Survived'])
    print(f"\nAccuracy: {accuracy:.4f}\n")


def cross_val_test(features):
    X_train, y_train = train_data(features)

    kf = KFold(n_splits=10, shuffle=True)

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
    diff = max(accuracy_scores) - min(accuracy_scores)
    print(f"Mean cross-validation accuracy: {mean_accuracy:.4f}, diff: {diff:.4f}")
    for i in accuracy_scores:
        print(f"{i:.4f}", end=' ')

def combinations_test():
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']
    X_train, y_train = train_data(features)
    kf = KFold(n_splits=10, shuffle=True)

    for comb in combinations(features, 5):
        X_train_comb = X_train[list(comb)]
        accuracy_scores = []

        for train_idx, val_idx in kf.split(X_train_comb):
            X_train_fold, X_val_fold = X_train_comb.iloc[train_idx], X_train_comb.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

            nb = NaiveBayesClassifier()
            nb.fit(X_train_fold, y_train_fold)
            predictions = nb.predict(X_val_fold)

            accuracy = accuracy_score(y_val_fold, predictions)
            accuracy_scores.append(accuracy)

        mean_accuracy = np.mean(accuracy_scores)
        diff = max(accuracy_scores) - min(accuracy_scores)
        print(f"\n\n{comb}: {mean_accuracy:.4f}, diff: {diff:.4f}")
        for i in accuracy_scores:
            print(f"{i:.4f}", end=' ')

if __name__ == "__main__":
    features_list = [
        ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'],
        ['Sex', 'Age', 'Fare', 'Parch', 'Embarked'],
        ['Age', 'Fare', 'SibSp', 'Parch', 'Embarked']
    ]


    for features in features_list:
        print(features)
        cross_val_test(features)
        key_test(features)

    # combinations_test()
