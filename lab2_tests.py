from lab2 import evolution_strategy


def sigma_tests():
    sigma_values = [0.01, 0.1, 1, 10]
    for sigma in sigma_values:
        evolution_strategy(
            128,
            512,
            sigma,
            1000,
            [3, 4],
            [-3, -4],
            max=False,
            chart=True
        )


def mi_lambda_tests():
    mi_lambda_values = [
        (1, 1),
        (1, 16),
        (16, 1),
        (16, 16),
        (128, 512)
    ]

    for mi, lmb in mi_lambda_values:
        evolution_strategy(
            mi,
            lmb,
            0.4,
            1000,
            [3, 4],
            [-3, -4],
            max=False,
            chart=True
        )


def custom_test():
    try:
        mi_size = int(input("Enter μ: "))
        lambda_size = int(input("Enter λ: "))
        sigma = float(input("Enter σ: "))
        generations = int(input("Enter number of generations: "))
        x_min = float(input("Enter minimum value for x range: "))
        x_max = float(input("Enter maximum value for x range: "))
        y_min = float(input("Enter minimum value for y range: "))
        y_max = float(input("Enter maximum value for y range: "))
        max_input = input("Search for max or min? (max/min): ").strip().lower()
        plot_input = input("Display chart? (y/n): ").strip().lower()

        max = True if max_input == "max" else False
        plot = True if plot_input == "y" else False

        evolution_strategy(
            mi_size,
            lambda_size,
            sigma,
            generations,
            [x_min, x_max],
            [y_min, y_max],
            max,
            plot
        )
    except ValueError:
        print("Invalid input.")


if __name__ == "__main__":
    # sigma_tests()
    # mi_lambda_tests()
    # custom_test()
    evolution_strategy(
        128,
        512,
        3,
        1000,
        [15, 15],
        [15, 15],
        max=False,
        chart=True
    )
