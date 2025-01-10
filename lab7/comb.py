import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from main import NaiveBayesClassifier, embarked_to_num, age_to_category, preprocess_data


if __name__ == "__main__":
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']

    data = pd.read_csv("train.csv", header=0)
    data = data[features + ['Survived']]
    data = preprocess_data(data)

    X_train = data.drop('Survived', axis='columns')
    y_train = data.Survived

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