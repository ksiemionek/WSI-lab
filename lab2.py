import matplotlib.pyplot as plt
import numpy as np


A = 1   # first non-zero digit after decimal point in sqrt(3.14 * 331430)


def function(x, y):
    return (A * x * y) / np.exp(x ** 2 + 0.5 * x + y ** 2)


def init_population(population_size, x_range, y_range):
    x_values = np.random.uniform(x_range[0], x_range[1], population_size)   # random values from the given x range
    y_values = np.random.uniform(y_range[0], y_range[1], population_size)   # random values from the given y range
    return np.column_stack((x_values, y_values))


def show_chart(all_points, best, mi_size, lambda_size, sigma):
    X, Y = np.meshgrid(np.arange(-3, 3, 0.05), np.arange(-3, 3, 0.05))
    plt.contour(X, Y, function(X, Y), levels=30, cmap="gray", zorder=1)
    plt.title(f"Test parameters: μ = {mi_size}, λ = {lambda_size}, σ = {sigma}\n" +
              f"Result - x: {best[0]:.5f}, y: {best[1]:.5f}, z: {function(best[0], best[1]):.5f}")
    all_points = np.array(all_points)
    plt.scatter(all_points[:, 0], all_points[:, 1], color="red", s=1, alpha=0.3)
    plt.scatter(best[0], best[1], color="green", s=10)
    plt.show()


def evolution_strategy(
        mi_size,
        lambda_size,
        sigma,
        generations,
        x_range,
        y_range,
        max=True,
        chart=False
):
    # initializing the population
    population = init_population(mi_size, x_range, y_range)
    all_points = []
    best = None

    for gen in range(generations):
        if chart: all_points.extend(population.tolist())

        # selecting parents for crossover
        parents1 = population[np.random.choice(mi_size, lambda_size)]
        parents2 = population[np.random.choice(mi_size, lambda_size)]

        # crossover
        distributions = np.random.uniform(0, 1, (lambda_size, 2))
        children = parents1 * distributions + parents2 * (1 - distributions)

        # adding gaussian noise (mutation)
        gaussian_noise = np.random.normal(0, sigma, (lambda_size, 2))
        children += gaussian_noise

        # stacking parents and children
        new_population = np.vstack((population, children))

        # calculating population values
        x_values = new_population[:, 0]
        y_values = new_population[:, 1]
        z_values = np.array(function(x_values, y_values))

        # sorting values
        idx_sorted = np.argsort(z_values)

        # searching for maximum or minimum
        if max:
            population = new_population[idx_sorted][-mi_size:]
            new_best = population[-1]
            if best is None or function(best[0], best[1]) < function(new_best[0], new_best[1]):
                best = new_best
        else:
            population = new_population[idx_sorted][:mi_size]
            new_best = population[0]
            if best is None or function(best[0], best[1]) > function(new_best[0], new_best[1]):
                best = new_best

    # displaying test parameters and result
    print(f"Test parameters: μ = {mi_size}, λ = {lambda_size}, σ = {sigma}")
    print(f"Population initialized within range {x_range} × {y_range}")
    print(f"Number of generations: {generations}")
    print(f"Result - x: {best[0]:.5f}, y: {best[1]:.5f}, z: {function(best[0], best[1]):.5f}\n")

    # optional chart showing the whole process and result
    if chart:
        show_chart(all_points, best, mi_size, lambda_size, sigma)
