import random
import matplotlib.pyplot as plt
import pygame
import torch
import numpy as np

from food import Food
from snake import Snake, Direction


def train_and_test(epsilon, move, apple, death, train=True):
    filename = f"q_eps{epsilon}:move{move}:apple{apple}:death{death}.tensor"

    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds)

    if train:
        agent = QLearningAgent(block_size, bounds, epsilon=epsilon, discount=0.99, is_training=True)
    else:
        agent = QLearningAgent(
            block_size,
            bounds,
            epsilon=0,
            discount=0.99,
            is_training=False,
            load_qfunction_path=filename
        )

    scores = []
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode = 0
    total_episodes = 1000 if train else 100
    while episode < total_episodes and run:
        # pygame.time.delay(1)  # Adjust game speed, decrease to learn agent faster

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state, reward, is_terminal)
        reward = -move
        is_terminal = False
        snake.turn(direction)
        snake.move()
        reward += snake.check_for_food(food) * apple
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(1)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()
            episode += 1
            reward -= death
            is_terminal = True

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    if train:
        plot_results(scores)
        agent.save_qfunction(filename)

    print(f"Scores: {scores}")
    print(f"Average score: {np.average(scores)}")

    pygame.quit()

    if train:
        for i in range(10):
            train_and_test(epsilon, move, apple, death, train=False)


def plot_results(scores):
    episodes = list(range(len(scores)))
    avg_scores = [sum(scores[:i+1]) / (i+1) for i in range(len(scores))]

    plt.figure(figsize=(9, 6))
    plt.plot(episodes, scores, label="Episode score")
    plt.plot(episodes, avg_scores, label="Average score")
    plt.title("Agent's training performance")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()


class QLearningAgent:
    def __init__(self, block_size, bounds, epsilon, discount, is_training=True, load_qfunction_path=None):
        """ You can change whatever you want."""
        self.block_size = block_size
        self.bounds = bounds
        self.epsilon = epsilon
        self.discount = discount
        self.is_training = is_training
        self.learning_rate = 0.1
        observation_shape = (2, 2, 2, 2, 2, 2, 2, 2, 4)
        self.Q = torch.zeros((*observation_shape, 4))  # There are 4 actions
        if not is_training:
            self.load_qfunction(load_qfunction_path)

        self.prev_obs = None
        self.prev_action = None

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state, reward, is_terminal)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        new_obs = self.game_state_to_observation(game_state)

        if self.prev_obs is not None:
            prev_q = self.Q[self.prev_obs][self.prev_action]
            max_future_q = torch.max(self.Q[new_obs])
            self.Q[self.prev_obs][self.prev_action] = prev_q +self.learning_rate * (
                    reward + self.discount * max_future_q - prev_q
            )

        if random.random() < self.epsilon:
            new_action = random.randint(0, 3)
        else:
            new_action = int(torch.argmax(self.Q[new_obs]))

        self.prev_obs = new_obs
        self.prev_action = new_action

        return Direction(int(new_action))

    def game_state_to_observation(self, game_state):
        size = self.block_size
        body = game_state["snake_body"]
        head = body[-1]
        food = game_state["food"]
        return (
                int((head[0], head[1] - size) in body or head[1] - size < 0),           # obstacle_up        (2)
                int((head[0] + size, head[1]) in body or head[0] + size >= self.bounds[0]),  # obstacle_right     (2)
                int((head[0], head[1] + size) in body or head[1] + size >= self.bounds[1]),  # obstacle_down      (2)
                int((head[0] - size, head[1]) in body or head[0] - size < 0),           # obstacle_left      (2)
                int(food[1] < head[1]),                                                 # food_up            (2)
                int(food[0] > head[0]),                                                 # food_right         (2)
                int(food[1] > head[1]),                                                 # food_down          (2)
                int(food[0] < head[0]),                                                 # food_left          (2)
                game_state["snake_direction"].value                                     # snake_direction    (4)
               )

    def act_test(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        new_obs = self.game_state_to_observation(game_state)
        new_action = int(torch.argmax(self.Q[new_obs]))
        return Direction(int(new_action))

    def save_qfunction(self, path):
        torch.save(self.Q, path)

    def load_qfunction(self, path):
        self.Q = torch.load(path)


if __name__ == "__main__":
    train_and_test(0.1, 1.0, 10.0, 25.0, train=True)
