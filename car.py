import numpy as np


class Car:
    def __init__(self, initial_action, epsilon=None, q_function=None):
        self.initial_action = initial_action
        self.current_action = initial_action
        self.epsilon = epsilon
        self.q_function = q_function
        self.possible_actions = {(-1, -1), (-1, 0), (0, 0), (0, -1), (0, 1), (1, 0), (1, 1)}

    def set_q_function(self, function):
        self.q_function = function

    def set_action(self, new_action):
        self.current_action = new_action

    def take_action(self, state=None):
        # Choose among next possible state-action pairs
        optimal_action, optimal_value = None, np.inf
        for action in self.possible_actions:
            value = self.q_function[(state, action)]
            if value < optimal_value:
                optimal_value = value
                optimal_action = action
        # Choose action probabilistically
        action_map = {
            0: optimal_action,
            1: (-1, -1),
            2: (-1, 0),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, 0),
            7: (1, 10)
        }
        probability = self.epsilon/6.0
        probabilities = [1 - self.epsilon] + [probability for _ in range(6)]
        taken_action = np.random.choice([_ for _ in range(7)], p=probabilities)
        self.current_action = action_map[taken_action]
        self.epsilon -= 0.001
        self.epsilon = max(self.epsilon, 0.01)
        return self.current_action

    def get_current_action(self):
        return self.current_action

    def get_all_actions(self):
        return self.possible_actions
