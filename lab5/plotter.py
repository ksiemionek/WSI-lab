from model import *
import matplotlib.pyplot as plt


def plot_results(train_results, val_results, test_results, neurons):
    plt.figure(figsize=(10, 6))

    plt.plot(neurons, train_results, label="Training Data", marker="o")
    plt.plot(neurons, val_results, label="Validation Data", marker="o")
    plt.plot(neurons, test_results, label="Test Data", marker="o")

    plt.xlabel("Number of Neurons")
    plt.ylabel("Performance")
    plt.title("Performance vs Number of Neurons")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_plot_high.png")
    plt.show(block=True)


def train_model_with_metrics(model, train_loader, val_loader, iterations, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_accuracies = []
    val_accuracies = []
    losses = []

    for iteration in range(iterations):
        model.train()
        correct_train = 0
        total_train = 0
        iteration_loss = 0.0

        for data, labels in train_loader:
            optimizer.zero_grad()
            results = model(data)
            loss = criterion(results, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(results, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            iteration_loss += loss.item()

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        losses.append(iteration_loss / len(train_loader))

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for data, labels in val_loader:
                results = model(data)
                _, predicted = torch.max(results, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = 100 * correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f"Iteration: {iteration + 1}, Loss: {iteration_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

    return train_accuracies, val_accuracies, losses

def compare_activation_functions(train_loader, val_loader, input_size, output_size, hidden_layers, iterations=30, learning_rate=0.1):
    activations = {
        "Identity": nn.Identity,
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU
    }

    results = {}

    for name, activation in activations.items():
        print(f"Training with activation function: {name}")
        model = MLP(input_size, hidden_layers, output_size, activation_fun=activation)
        train_accuracies, val_accuracies, losses = train_model_with_metrics(
            model, train_loader, val_loader, iterations, learning_rate
        )
        results[name] = {
            "train_accuracy": train_accuracies,
            "val_accuracy": val_accuracies,
            "loss": losses
        }

    plt.figure(figsize=(8, 6))
    for name, metrics in results.items():
        plt.plot(metrics["val_accuracy"], label=f"{name} (Validation)")
        plt.plot(metrics["train_accuracy"], linestyle='--', label=f"{name} (Training)")

    plt.title("Activation Functions")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    for name, metrics in results.items():
        plt.plot(metrics["loss"], label=f"{name}")

    plt.title("Activation Functions")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def get_gradient_norms(model):
    norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            norms.append(param.grad.norm().item())
    return norms

def train_and_get_gradients(model, train_loader, learning_rate=0.1, num_epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    gradient_norms = []

    for epoch in range(num_epochs):
        model.train()
        epoch_norms = []

        for data, labels in train_loader:
            optimizer.zero_grad()
            results = model(data)
            loss = criterion(results, labels)
            loss.backward()
            optimizer.step()

            epoch_norms.append(torch.tensor(get_gradient_norms(model)))

        gradient_norms.append(torch.stack(epoch_norms).mean(dim=0))

    return gradient_norms

def plot_gradient_norms(gradient_norms):
    average_norms = np.mean(gradient_norms, axis=0)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(average_norms)), average_norms, marker='o')
    plt.title("Gradient norms")
    plt.xlabel("Layer")
    plt.ylabel("Gradient norm")
    plt.grid()
    plt.show()

def gradient_norms_result():
    dataset_path = "./data/training_data_new_skuter.pt"
    dataset = BCDataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)

    input_size = dataset.data.shape[1]
    output_size = 4
    hidden_layers = [32] * 30

    model = MLP(input_size, hidden_layers, output_size, activation_fun=nn.ReLU)

    gradient_norms = train_and_get_gradients(model, train_loader, num_epochs=1)

    plot_gradient_norms(gradient_norms)


def compare_activation_functions_result(layers):
    dataset_path = "./data/training_data_new_skuter.pt"
    dataset = BCDataset(dataset_path)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    input_size = dataset.data.shape[1]
    output_size = 4
    hidden_layers = [32] * layers

    compare_activation_functions(train_loader, val_loader, input_size, output_size, hidden_layers)


if __name__ == "__main__":
    neuron_count = [32, 64, 128, 512, 1024]
    results_train = []
    results_test = []
    results_val = []
    for count in neuron_count:
        set = BCDataset("./data/training_data_new_skuter.pt")
        train_size = int(0.8 * len(set))
        val_size = int(0.1 * len(set))
        test_size = len(set) - train_size - val_size

        train_set, val_set, test_set = random_split(set, [train_size, val_size, test_size])

        train_loader = DataLoader(train_set, 64, True, num_workers=0)
        val_loader = DataLoader(val_set, 64, True, num_workers=0)
        test_loader = DataLoader(test_set, 64, True, num_workers=0)

        input_size = train_set.dataset.data.shape[1]

        model = MLP(input_size, hidden_size=[count], output_size=4)

        result_train, result_val = train_model(model, train_loader, val_loader, 100, 0.1)
        result_test = test_model(model, test_loader)
        results_train.append(result_train)
        results_val.append(result_val)
        results_test.append(result_test)
    plot_results(results_train, results_val, results_test, neuron_count)
    print("Training Results:", results_train)
    print("Validation Results:", results_val)
    print("Test Results:", results_test)
