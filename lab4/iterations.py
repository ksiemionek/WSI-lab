import copy
import os
import pickle

import numpy as np
import pygame
import time

from food import Food
from lab4.model import LogisticRegressionModel
from model import game_state_to_data_sample
from model import files_to_data
from snake import Snake, Direction


iterations_scores = []

def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    games = 100

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    for i in range(0, 5000, 250):
        agent = BehavioralCloningAgent(block_size, bounds, 1, i)  # Once your agent is good to go, change this line
        scores = []
        run = True
        pygame.time.delay(1000)
        while run:
            pygame.time.delay(2)  # Adjust game speed, decrease to test your agent and model quickly

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
                pygame.time.delay(200)
                scores.append(snake.length - 3)
                games -= 1
                if games == 0:
                    avg = np.average(scores)
                    iterations_scores.append(avg)
                    print(f"Average score for {i} iterations: {avg}")
                    games = 100
                    break
                snake.respawn()
                food.respawn()

            window.fill((0, 0, 0))
            snake.draw(pygame, window)
            food.draw(pygame, window)
            pygame.display.update()


class BehavioralCloningAgent:
    def __init__(self, block_size, bounds, learning_rate, iterations):
        self.block_size = block_size
        self.bounds = bounds
        self.model = LogisticRegressionModel(learning_rate, iterations)

        X, y = files_to_data('test')

        print(X.shape)

        self.model.fit(X, y)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        game_state_features = game_state_to_data_sample(game_state, self.block_size, self.bounds)
        move = self.model.predict(game_state_features)
        return Direction(move)

    def dump_data(self):
        pass


if __name__ == "__main__":
    main()
    print(iterations_scores)

