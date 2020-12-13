import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class QLearning:
    """
    Class that implements Q-Learning algorithm
    """
    def __init__(self, discount, learning_rate, threshold, max_iterations, environment, agent):
        """
        Initializes class
        :param discount: Float
        :param learning_rate: Float
        :param threshold: Float
        :param max_iterations: Int
        :param environment: RaceTrack
        :param agent: Car
        """
        self.discount = discount
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.environment = environment
        self.agent = agent
        self.avg_costs = []

    def __initialize_values(self):
        """
        Initializes q function
        :return: None
        """
        self.all_states = self.environment.get_all_states()
        self.all_actions = self.agent.get_all_actions()
        self.state_action_pairs = {(s, a) for s in self.all_states for a in self.all_actions}
        self.q_function = {k: v for (k, v) in zip(self.state_action_pairs, np.zeros(len(self.state_action_pairs)))}
        self.agent.set_q_function(self.q_function)

    def run(self):
        """
        Runs algorithm
        :return: List
        """
        self.__initialize_values()
        q = deque(maxlen=25)
        cost = np.inf
        i = 0
        num_iterations = 0
        path = []
        while cost > self.threshold and i < self.max_iterations:
            initial_state = self.environment.reset_state()
            current_state = initial_state
            path = [current_state]
            while not self.environment.in_terminal_state():
                current_action = self.agent.take_action(current_state)
                next_state = self.environment.update_state(current_action, indicate_random=True)
                reward = self.environment.get_reward(next_state)
                next_action = self.agent.take_action(next_state)
                current_q = self.q_function[(current_state, current_action)]
                next_q = self.q_function[(next_state, next_action)]
                self.q_function[(current_state, current_action)] += self.learning_rate * (reward + self.discount * next_q - current_q)
                print(current_state, next_state, current_action, '\n')
                current_state = next_state
                path.append(current_state)
            self.learning_rate -= 0.0001
            self.learning_rate = max(self.learning_rate, 0.001)
            q.appendleft(len(path))
            if len(q) == 25:
                cost = sum(q) / len(q)
                print("Average cost: ", cost)
                self.avg_costs.append(cost)
                q.pop()
            i += 1
            num_iterations += 1
            print("Iterations: ", num_iterations, '--------------------------------------------')
        print("Finished training---------------------------------------")
        return path

    def plot_avg_cost(self):
        """
        Plot average cost over time
        :return: None
        """
        plt.plot(self.avg_costs)
        plt.ylabel('Average Cost')
        plt.savefig("q_avg_cost_graph.png")
