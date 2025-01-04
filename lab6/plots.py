import random
import matplotlib.pyplot as plt
import pygame
import torch
import numpy as np

from food import Food
from snake import Snake, Direction
from main_rl import QLearningAgent


def plot_rewards(epsilon_values, rewards_dict):
    plt.figure(figsize=(9, 6))
    for epsilon in epsilon_values:
        rewards = rewards_dict[epsilon]
        smoothed_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label=f"ε={epsilon}")

    plt.title("Sum of Rewards per Episode for Different ε Values")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_and_test_with_rewards(epsilon, move, apple, death, version, train=True):
    filename = f"q_eps{epsilon}:move{move}:apple{apple}:death{death}:v{version}.tensor"

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
    rewards = []
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode = 0
    total_episodes = 1000 if train else 100
    cumulative_reward = 0

    while episode < total_episodes and run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,
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
            cumulative_reward += reward
            rewards.append(cumulative_reward)
            cumulative_reward = 0
            snake.respawn()
            food.respawn()
            episode += 1
            reward -= death
            is_terminal = True
        else:
            cumulative_reward += reward

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    if train:
        agent.save_qfunction(filename)

        print(f"Scores: {scores}")
        print(f"Average score (training, version {version}): {np.average(scores)}")
    else:
        print(f"Average score (testing): {np.average(scores)}")

    pygame.quit()

    return rewards


if __name__ == "__main__":
    epsilon_values = [0.01, 0.1, 0.5]
    all_rewards = {}

    for epsilon in epsilon_values:
        rewards = train_and_test_with_rewards(epsilon, 0.1, 1.2, 2.5, version=1, train=True)
        all_rewards[epsilon] = rewards

    plot_rewards(epsilon_values, all_rewards)