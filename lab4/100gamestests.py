import numpy as np
import pygame

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from food import Food
from lab4.model import LogisticRegressionModel
from model import game_state_to_data_sample
from model import files_to_data
from snake import Snake, Direction


# rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# iterations = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

rates = [10]
iterations = [100]
GAMES = 100
plot = False

rates_scores = []


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    games = GAMES

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    for rate in rates:
        rate_scores = []
        print(f"LEARNING RATE = {rate}")

        for iter in iterations:
            agent = BehavioralCloningAgent(block_size, bounds, rate, iter)  # Once your agent is good to go, change this line
            scores = []
            run = True
            pygame.time.delay(1000)
            while run:
                pygame.time.delay(10)  # Adjust game speed, decrease to test your agent and model quickly

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

                game_state = {"food": (food.x, food.y),
                              "snake_body": snake.body,  # The last element is snake's head
                              "snake_direction": snake.direction}

                direction = agent.act(game_state)
                snake.turn(direction)

                snake.move()
                snake.check_for_food(food)
                food.update()

                if snake.is_wall_collision() or snake.is_tail_collision():
                    pygame.display.update()
                    pygame.time.delay(300)
                    scores.append(snake.length - 3)
                    games -= 1
                    if games == 0:
                        avg = np.average(scores)
                        rate_scores.append(avg)
                        print(f"Scores: {scores}")
                        print(f"Average score for {iter} iterations: {avg}")
                        games = GAMES
                        break
                    snake.respawn()
                    food.respawn()

                window.fill((0, 0, 0))
                snake.draw(pygame, window)
                food.draw(pygame, window)
                pygame.display.update()

        rates_scores.append(rate_scores)


class BehavioralCloningAgent:
    def __init__(self, block_size, bounds, learning_rate, iterations):
        self.block_size = block_size
        self.bounds = bounds
        self.model = LogisticRegressionModel(learning_rate, iterations)

        X, y = files_to_data('test')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        game_state_features = game_state_to_data_sample(game_state, self.block_size, self.bounds)
        move = self.model.predict(game_state_features)
        return Direction(move)


if __name__ == "__main__":
    main()
    if plot:
        for i in range(len(rates)):
            plt.plot(iterations, rates_scores[i], label="average score")
        plt.legend()
        plt.show()