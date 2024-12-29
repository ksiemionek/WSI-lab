import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from snake import Direction

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""


def game_state_to_data_sample(game_state: dict, bounds: tuple, block_size: int):
    bound_x, bound_y = bounds
    food = game_state["food"]
    head = game_state["snake_body"][-1]
    data_sample = []
    data_sample.append(0 if food[1] > head[1] else 1)
    data_sample.append(1 if food[0] > head[0] else 0)
    data_sample.append(1 if food[1] > head[1] else 0)
    data_sample.append(0 if food[0] > head[0] else 1)
    future_head = (head[0], head[1] - block_size)
    if future_head in game_state["snake_body"] or future_head[0] < 0:
        data_sample.append(1)
    else:
        data_sample.append(0)
    future_head = (head[0] + block_size, head[1])
    if future_head in game_state["snake_body"] or future_head[0] >= bound_x:
        data_sample.append(1)
    else:
        data_sample.append(0)
    future_head = (head[0], head[1] + block_size)
    if future_head in game_state["snake_body"] or future_head[1] >= bound_y:
        data_sample.append(1)
    else:
        data_sample.append(0)
    future_head = (head[0] - block_size, head[1])
    if future_head in game_state["snake_body"] or future_head[0] < 0:
        data_sample.append(1)
    else:
        data_sample.append(0)
    parsed_dir=parse_dir(game_state["snake_direction"])
    for i in range(4):
        data_sample.append(0 if i!=parsed_dir else 1)
    return torch.FloatTensor(data_sample)


def parse_dir(dir):
    match dir:
        case Direction.UP:
            return 0
        case Direction.RIGHT:
            return 1
        case Direction.DOWN:
            return 2
        case Direction.LEFT:
            return 3

def generate_data():
    data_folder = "./data/dataToProcess"
    all_data = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".pickle"):
            with open(os.path.join(data_folder, filename), "rb") as f:
                data_file = pickle.load(f)
            for state in data_file["data"]:
                new_data = torch.cat(
                    (
                        game_state_to_data_sample(
                            state[0], data_file["bounds"], data_file["block_size"]
                        ),
                        torch.DoubleTensor([parse_dir(state[1])]),
                    ),
                    dim=0,
                )
                all_data.append(new_data)
    all_data = torch.stack(all_data)
    print(all_data, all_data.size())
    torch.save(all_data, "./data/training_data_new_skuter.pt")


def files_to_data(directory):
    all_x = []
    all_y = []
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            with open(f"{directory}/{filename}", "rb") as f:
                data_file = pickle.load(f)
                block_size = data_file["block_size"]
                bounds = data_file["bounds"]
                for game_state, direction in data_file["data"]:
                    data = game_state_to_data_sample(game_state, block_size, bounds)
                    all_x.append(data)
                    all_y.append(direction.value)
    return np.array(all_x), np.array(all_y)


class BCDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=True)
        self.data = data[:, :-1].float()
        self.labels = data[:, -1].long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation_fun=nn.ReLU,
    ):
        super(MLP, self).__init__()

        layers = []

        prev_size = input_size

        # HIDDEN LAYERS
        for size in hidden_size:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation_fun())
            layers.append(nn.Dropout())
            prev_size = size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def save_state_dict(self):
        torch.save(self.model.state_dict(), "./data/state_dict.pth")

    def predict_direction(self, data):
        self.eval()
        with torch.no_grad():
            results = self(data)
            _, predicted = torch.max(results.data, 0)
        return Direction(predicted.item())

    @classmethod
    def from_state_dict(cls, state_dict_path, input_size, hidden_size, output_size, activation_fun=nn.ReLU):
        model = cls(input_size, hidden_size, output_size, activation_fun)
        model.model.load_state_dict(torch.load(state_dict_path))
        return model


def train_model(model: MLP, train_loader, val_loader, iterations, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate)
    best_accuracy, best_weights = 0, None
    for iteration in range(iterations):
        model.train()
        iteration_loss = 0.0
        correct_train = 0
        total_train = 0

        for data, labels in train_loader:
            results = model(data)
            loss = criterion(results, labels)
            optimizer.zero_grad()  # reset optimizer
            loss.backward()  # backpropagation
            optimizer.step()

            iteration_loss += loss.item()
            _, predicted = torch.max(results.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                results = model(data)
                _, predicted = torch.max(results.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        val_accuracy = 100 * correct_val / total_val

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            model.save_state_dict()
        print(
            f"Iteration: {iteration+1}, Loss: {iteration_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%"
        )

    return train_accuracy, val_accuracy


def test_model(model: MLP, test_loader):
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, labels in test_loader:
            results = model(data)
            _, predicted = torch.max(results.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy


if __name__ == "__main__":
    set = BCDataset("./data/training_data_new_skuter.pt")
    # loader = DataLoader(set, 64, True, num_workers=0)
    # for data, labels in loader:
    #     print(data.size(), labels.size())

    train_size = int(0.8 * len(set))
    val_size = int(0.1 * len(set))
    test_size = len(set) - train_size - val_size

    train_set, val_set, test_set = random_split(set, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, 64, True, num_workers=0)
    val_loader = DataLoader(val_set, 64, True, num_workers=0)
    test_loader = DataLoader(test_set, 64, True, num_workers=0)

    input_size = train_set.dataset.data.shape[1]

    model = MLP(input_size, hidden_size=[128 for _ in range(5)], output_size=4)

    train_model(model, train_loader, val_loader, 100, 0.1)
    test_model(model, test_loader)
    state_dict = model.model.state_dict()
    model.save_state_dict()
    generate_data()

