import pandas as pd


def discretize_age(age):
    if pd.isna(age):
        return 'unknown'
    elif age <= 20:
        return 'child'
    elif age <= 60:
        return 'adult'
    else:
        return 'elderly'


def discretize_fare(fare):
    if pd.isna(fare):
        return 'unknown'
    elif fare <= 10:
        return 'low'
    elif fare <= 30:
        return 'medium'
    elif fare <= 100:
        return 'high'
    else:
        return 'very high'


def discretize_sibsp(sibsp):
    if pd.isna(sibsp):
        return 'unknown'
    elif sibsp == 0:
        return 'none'
    elif sibsp <= 2:
        return 'few'
    elif sibsp <= 4:
        return 'more'
    else:
        return 'many'


def discretize_parch(parch):
    if pd.isna(parch):
        return 'unknown'
    elif parch == 0:
        return 'none'
    elif parch <= 2:
        return 'few'
    else:
        return 'many'


def preprocess_data(data):
    data['Age'] = data['Age'].apply(discretize_age)
    data['Fare'] = data['Fare'].apply(discretize_fare)
    data['SibSp'] = data['SibSp'].apply(discretize_sibsp)
    data['Parch'] = data['Parch'].apply(discretize_parch)

    return data


def train_data(features):
    data = pd.read_csv("train.csv", header=0)
    data = preprocess_data(data)
    data = data[features + ['Survived']]

    X_train = data.drop('Survived', axis='columns')
    y_train = data.Survived

    return X_train, y_train

def test_data(features):
    test = pd.read_csv("test.csv", header=0)
    test = preprocess_data(test)

    return test[features]
