import pickle
import numpy as np
import os

from snake import Direction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""


def is_barrier(position, bounds, snake_body):
    x, y = position
    if x < 0 or x >= bounds[0] or y < 0 or y >= bounds[1]:
        return 1
    if position in snake_body:
        return 1
    return 0


def is_food(position, food_position):
    return 1 if position == food_position else 0


def food_distance_in_direction(snake_head, food_position, block_size, direction):
    x, y = snake_head
    food_x, food_y = food_position

    # up
    if direction == 0:
        return max(0, y - food_y) / block_size
    # right
    elif direction == 1:
        return max(0, food_x - x) / block_size
    # down
    elif direction == 2:
        return max(0, food_y - y) / block_size
    # left
    else:
        return max(0, x - food_x) / block_size


def food_in_direction(snake_head, food_position, direction):
    x, y = snake_head
    food_x, food_y = food_position

    # up
    if direction == 0:
        return 1 if food_x == x and max(0, y - food_y) else 0
    # right
    elif direction == 1:
        return 1 if food_y == y and max(0, food_x - x) else 0
    # down
    elif direction == 2:
        return 1 if food_x == x and max(0, food_y - y) else 0
    # left
    else:
        return 1 if food_y == y and max(0, x - food_x) else 0


def game_state_to_data_sample(game_state: dict, block_size, bounds):
    food_position = game_state["food"]
    snake_body = game_state["snake_body"][:-1]
    snake_head = game_state["snake_body"][-1]
    direction = game_state["snake_direction"]

    x, y = snake_head
    up = (x, y - block_size)
    down = (x, y + block_size)
    left = (x - block_size, y)
    right = (x + block_size, y)

    result = [
        is_barrier(up, bounds, snake_body),
        is_barrier(right, bounds, snake_body),
        is_barrier(down, bounds, snake_body),
        is_barrier(left, bounds, snake_body),
        # is_food(up, food_position),
        # is_food(right, food_position),
        # is_food(down, food_position),
        # is_food(left, food_position),
        direction.value == Direction.UP.value,
        direction.value == Direction.RIGHT.value,
        direction.value == Direction.DOWN.value,
        direction.value == Direction.LEFT.value,
        # food_distance_in_direction(snake_head, food_position, block_size, 0),
        # food_distance_in_direction(snake_head, food_position, block_size, 1),
        # food_distance_in_direction(snake_head, food_position, block_size, 2),
        # food_distance_in_direction(snake_head, food_position, block_size, 3)
        food_in_direction(snake_head, food_position, 0),
        food_in_direction(snake_head, food_position, 1),
        food_in_direction(snake_head, food_position, 2),
        food_in_direction(snake_head, food_position, 3)
    ]

    return np.array(result)


def files_to_data(directory):
    all_x = []
    all_y = []
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            with open(f"{directory}/{filename}", 'rb') as f:
                data_file = pickle.load(f)
                block_size = data_file["block_size"]
                bounds = data_file["bounds"]
                for game_state, direction in data_file["data"]:
                    data = game_state_to_data_sample(game_state, block_size, bounds)
                    all_x.append(data)
                    all_y.append(direction.value)
    return np.array(all_x), np.array(all_y)


class LogisticRegressionModel:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        x = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def move_prob(self, move_values, weights, bias):
        z = np.dot(weights, move_values) + bias
        return self.sigmoid(z)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros((4, num_features))
        self.bias = np.zeros(4)

        for _ in range(self.iterations):
            for i in range(4):
                labels = np.array([1 if label == i else 0 for label in y])
                scores = np.dot(X, self.weights[i]) + self.bias[i]
                predictions = self.sigmoid(scores)
                error = predictions - labels

                dw = (1 / num_samples) * np.dot(X.T, error)
                db = (1 / num_samples) * np.sum(error)

                self.weights[i] -= self.learning_rate * dw
                self.bias[i] -= self.learning_rate * db

    def predict(self, X):
        predictions = []
        for i in range(4):
            scores = np.dot(X, self.weights[i]) + self.bias[i]
            prediction = self.sigmoid(scores)
            predictions.append(prediction)
        return np.argmax(predictions, axis=0)


def data_size_test():
    X, y = files_to_data('test')

    # print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LogisticRegressionModel(1, 3000)
    train_sizes = [0.01, 0.1, 0.8]

    for train_size in train_sizes:
        X_train2, _, y_train2, _ = train_test_split(X_train, y_train, train_size=train_size)
        model.fit(X_train2, y_train2)
        print(f"{train_size * 100:.0f}% of data")
        print(f"Model test accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
        print(f"Model train accuracy: {accuracy_score(y_train2, model.predict(X_train2)) * 100:.2f}%")

    model.fit(X_train, y_train)
    print(f"100% of data")
    print(f"Model test accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
    print(f"Model train accuracy: {accuracy_score(y_train, model.predict(X_train)) * 100:.2f}%")


def learning_rate_test():
    X, y = files_to_data('test')

    # print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rates = [0.001, 0.01, 0.1, 0.5, 1]
    train_scores = []
    test_scores = []

    for rate in rates:
        model = LogisticRegressionModel(rate, 3000)
        model.fit(X_train, y_train)
        test_score = accuracy_score(y_test, model.predict(X_test))
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_scores.append(test_score)
        train_scores.append(train_score)
        print(f"Learning rate = {rate}")
        print(f"Model test accuracy: {test_score * 100:.2f}%")
        print(f"Model train accuracy: {train_score * 100:.2f}%\n")

    plt.plot(rates, train_scores, label="Train accuracy")
    plt.plot(rates, test_scores, label="Test accuracy")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def iterations_test():
    X, y = files_to_data('test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_scores = []
    test_scores = []

    for i in range(0, 6000, 100):
        model = model = LogisticRegressionModel(1, i)
        model.fit(X_train, y_train)
        test_score = accuracy_score(y_test, model.predict(X_test))
        train_score = accuracy_score(y_train, model.predict(X_train))
        test_scores.append(test_score)
        train_scores.append(train_score)
        print(i)


    iterations = [i for i in range(0, 6000, 50)]
    plt.plot(iterations, train_scores, label="Train accuracy")
    plt.plot(iterations, test_scores, label="Test accuracy")
    plt.xlabel("Learning rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # data_size_test()
    # learning_rate_test()
    iterations_test()