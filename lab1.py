import numpy as np
import matplotlib.pyplot as plt


A = 1   # first non-zero digit after decimal point in sqrt(3.14 * 331430)
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]


def function(x, y):
    return (A * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)


def gradient_vector(x, y):
    return  ((A * y * (1 - 2 * x ** 2 - 0.5 * x)) / np.exp(x ** 2 + 0.5 * x + y ** 2),  # x derivative
            (A * x * (1 - 2 * y ** 2)) / np.exp(x ** 2 + 0.5 * x + y ** 2))             # y derivative


x_range = np.arange(-3, 3, 0.05)
y_range = np.arange(-3, 3, 0.05)

X, Y = np.meshgrid(x_range, y_range)
Z = function(X, Y)

# learning rate > 0 -  maximum
# learning rate < 0 - minimum
def gradient_descent_result(position, learning_rate, n, rate_test=True):
    x = position[0]
    y = position[1]

    x_path = [x]
    y_path = [y]

    for _ in range(n):
        x_derivative, y_derivative = gradient_vector(x, y)
        x = x + learning_rate * x_derivative
        y = y + learning_rate * y_derivative

        x_path.append(x)
        y_path.append(y)
    if rate_test is True:
        return x_path, y_path, np.abs(learning_rate)
    else:
        point = "(" + str(round(x_path[0], 3)) + ", " + str(round(y_path[0], 3)) + ")"
        return x_path, y_path, point


def text_result(paths_list):
    for paths in paths_list:
        x_path, y_path, path_label = paths
        z = function(x_path[-1], y_path[-1])
        print("Starting point - x: " + str(round(x_path[0], 3)) + ", y: " + str(round(y_path[0], 3)) + '\n'
                + "Result - x: " + str(round(x_path[-1], 3)) + ", y: " + str(round(y_path[-1], 3)) + ", z: " + str(round(z, 3)))
    print('\n')

def plot_result(paths_list): # max 7 paths
    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(121, projection="3d", computed_zorder=False)
    ax1.plot_surface(X, Y, Z, cmap="gray")

    ax2 = fig.add_subplot(122)
    ax2.contour(X, Y, Z, levels=30, cmap="gray")

    for idx, paths in enumerate(paths_list):
        x_path, y_path, path_label = paths
        if len(paths_list) <= 7:
            ax1.plot(x_path, y_path, function(np.array(x_path), np.array(y_path)), color=COLORS[idx], markersize=3, label=path_label)
            ax2.plot(x_path, y_path, color=COLORS[idx], markersize=3, label=path_label)
        else:
            ax1.plot(x_path, y_path, function(np.array(x_path), np.array(y_path)), color="red", markersize=3, label=path_label)
            ax2.plot(x_path, y_path, color="red", markersize=3, label=path_label)
    if len(paths_list) <= 7:
        ax2.legend()
    plt.show()


"""
def main():
    # grid test
    grid_positions = []
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for y in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            grid_positions.append((x, y, function(x, y)))

    grid_paths_list_max = []
    grid_paths_list_min = []
    for position in grid_positions :
        grid_paths_list_max.append(gradient_descent_result(position, 0.2, 1000, False))
        grid_paths_list_min.append(gradient_descent_result(position, -0.2, 1000, False))
    plot_result(grid_paths_list_max)
    plot_result(grid_paths_list_min)
    text_result(grid_paths_list_max)
    text_result(grid_paths_list_min)

    # first test
    max_positions = [
        (-1.2, -2.4, function(-1.2, -2.4)),
        (-2.5, -1.1, function(2.5, -1.1)),
        (-0.1, -0.1, function(-0.1, -0.1)),
        (0.1, 0.1, function(0.1, 0.1)),
        (0.5, 1.5, function(0.5, 1.5)),
        (2.5, 1.0, function(2.5, 1.0)),
        (1.2, -1.2, function(1.2, -1.2))
    ]

    paths_list_max = []
    for position in max_positions:
        paths_list_max.append(gradient_descent_result(position, 0.2, 1000, False))
    plot_result(paths_list_max)
    text_result(paths_list_max)

    # second test
    min_positions = [
        (-2.2, 2.1, function(-2.2, 2.1)),
        (-1.5, -0.5, function(-1.5, -0.5)),
        (0.5, 1.5, function(0.5, 1.5)),
        (1.2, -1.2, function(1.2, -1.2)),
        (-0.5, -2.0, function(-0.5, -2.0)),
        (0.5, -0.5, function(0.5, -0.5)),
        (-1.2, -1.2, function(-1.2, -1.2))
    ]

    paths_list_min = []
    for position in min_positions:
        paths_list_min.append(gradient_descent_result(position, -0.2, 1000, False))
    plot_result(paths_list_min)
    text_result(paths_list_min)

    # third test
    paths_list_rate_min = []
    learning_rates_min = [-10, -5, -2, -0.8, -0.4, -0.1, -0.01]
    for rate in learning_rates_min:
        paths_list_rate_min.append(gradient_descent_result((-0.5, -2.0, function(-0.5, -2.0)), rate, 1000))
    plot_result(paths_list_rate_min)
    text_result(paths_list_rate_min)

    # fourth test
    paths_list_rate_max = []
    learning_rates_max = [10, 5, 2, 0.8, 0.4, 0.1, 0.01]
    for rate in learning_rates_max:
        paths_list_rate_max.append(gradient_descent_result((-2.2, 0.5, function(-2.2, 0.5)), rate, 1000))
    plot_result(paths_list_rate_max)
    text_result(paths_list_rate_max)
"""

def main():
    path = [gradient_descent_result((10.0, 10.0), 10, 1000)]
    text_result(path)
    plot_result(path)

if __name__ == "__main__":
    main()