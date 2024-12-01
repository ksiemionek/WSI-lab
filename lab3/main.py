import random
import time
import numpy as np
from matplotlib import pyplot as plt


random.seed(8)  # For final results set seed as your student's id modulo 42


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__ (OOOO000O000O00000 ):
        OOOO000O000O00000 .numbers =[]
    def act (O000000O000OO0O0O ,O0OO0O0O0O0OO0O00 ):
        if len (O0OO0O0O0O0OO0O00 )%2 ==0 :
            O00O0O0000000OO0O =sum (O0OO0O0O0O0OO0O00 [::2 ])
            O0O00O0OO00O0O0O0 =sum (O0OO0O0O0O0OO0O00 )-O00O0O0000000OO0O
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
                return O0OO0O0O0O0OO0O00 [1 :]  # explained: https://blogs.mtdv.me/articles/quantum_decision_search
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
            return O0OO0O0O0O0OO0O00 [:-1 ]
        else :
            O00O0O0000000OO0O =max (sum (O0OO0O0O0O0OO0O00 [1 ::2 ]),sum (O0OO0O0O0O0OO0O00 [2 ::2 ]))
            O0O00O0OO00O0O0O0 =max (sum (O0OO0O0O0O0OO0O00 [:-1 :2 ]),sum (O0OO0O0O0O0OO0O00 [:-2 :2 ]))
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
                return O0OO0O0O0O0OO0O00 [:-1 ]
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
            return O0OO0O0O0O0OO0O00 [1 :]


class MinMaxAgent:
    def __init__(self, max_depth=50):
        self.numbers = []
        self.max_depth = max_depth

    def act(self, vector: list):
        _, best_move = self._minmax(vector, False, self.max_depth)
        if best_move == 0:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]

    def _minmax(self, vector, is_maximizing, depth):
        if depth == 0 or len(vector) == 1:
            if is_maximizing:
                return max(vector[-1], vector[0]), -1 if vector[-1] > vector[0] else 0
            return -max(vector[-1], vector[0]), -1 if vector[-1] > vector[0] else 0

        if is_maximizing:
            left_score, _ = self._minmax(vector[1:], False, depth-1)
            right_score, _ = self._minmax(vector[:-1], False, depth-1)
            left_score += vector[0]
            right_score += vector[-1]
            return max(left_score, right_score), -1 if right_score > left_score else 0
        else:
            left_score, _ = self._minmax(vector[1:], True, depth-1)
            right_score, _ = self._minmax(vector[:-1], True, depth-1)
            left_score -= vector[0]
            right_score -= vector[-1]
            return min(left_score, right_score), -1 if right_score < left_score else 0


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def main():
    vector = [random.randint(-10, 10) for _ in range(14)]
    print(f"Vector: {vector}")
    first_agent, second_agent = NinjaAgent(), GreedyAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n"
          f"Second agent: {second_agent.numbers}")


def run_games(agent1, agent2, games=1000, vector_size=15):
    results1 = []
    results2 = []
    times = []
    for game in range(games):
        vector = [random.randint(-10, 10) for _ in range(vector_size)]
        agent1.numbers = []
        agent2.numbers = []

        if game % 2 == 0:
            first_agent, second_agent = agent1, agent2
        else:
            first_agent, second_agent = agent2, agent1

        start_time = time.time()
        run_game(vector, first_agent, second_agent)
        final_time = time.time() - start_time

        times.append(final_time)
        if game % 2 == 0:
            results1.append(sum(first_agent.numbers))
            results2.append(sum(second_agent.numbers))
        else:
            results1.append(sum(second_agent.numbers))
            results2.append(sum(first_agent.numbers))

    return np.mean(times), np.mean(results1), np.std(results1), results1, np.mean(results2), np.std(results2), results2


def analyze_results(first_agent, second_agent, games=1000, vector_size=15):
    first_agent_name = str(first_agent.__class__.__name__)
    second_agent_name = str(second_agent.__class__.__name__)
    for depth in [1, 2, 3, 15]:
        if first_agent_name == "MinMaxAgent":
            first_agent.max_depth = depth
        if second_agent_name == "MinMaxAgent":
            second_agent.max_depth = depth
        avg_time, avg_score1, std_dev1, scores1, avg_score2, std_dev2, scores2 = run_games(first_agent, second_agent, games, vector_size)

        # Results presented in the table
        print(f"Depth: {depth}, {games} games, average time: {avg_time:.6f}s")
        print("+-------------+---------------+---------+")
        print("|    Agent    | Average score | Std dev |")
        print("+-------------+---------------+---------+")
        print(f"| {first_agent_name:>11} | {avg_score1:>13.2f} | {std_dev1:>7.2f} |")
        print("+-------------+---------------+---------+")
        print(f"| {second_agent_name:>11} | {avg_score2:>13.2f} | {std_dev2:>7.2f} |")
        print("+-------------+---------------+---------+\n")

        # Histogram for depth 2 and 15
        if depth in [2, 15]:
            plt.hist(scores1, bins=20, color="red", alpha=0.5, label=first_agent_name)
            plt.hist(scores2, bins=20, color="blue", alpha=0.5, label=second_agent_name)
            plt.title(f"Depth = {depth}, {games} games")
            plt.xlabel("Scores")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    analyze_results(GreedyAgent(), MinMaxAgent())
