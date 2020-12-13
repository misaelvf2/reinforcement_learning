import numpy as np


class Car:
    """
    Class for representing and abstracting Car agent
    """
    def __init__(self, initial_action, epsilon=None, q_function=None):
        """
        Initializes Car class
        :param initial_action: Tuple
        :param epsilon: Float
        :param q_function: Dict
        """
        self.initial_action = initial_action
        self.current_action = initial_action
        self.epsilon = epsilon
        self.q_function = q_function
        self.possible_actions = {(-1, -1), (-1, 0), (0, 0), (0, -1), (0, 1), (1, 0), (1, 1)}

    def set_q_function(self, function):
        """
        Sets q function to given function
        :param function: Dict
        :return: None
        """
        self.q_function = function

    def set_action(self, new_action):
        """
        Sets action to given action
        :param new_action: Tuple
        :return: None
        """
        self.current_action = new_action

    def take_action(self, state=None):
        """
        Changes car's current action to new action using epsilon-greedy search
        :param state: Tuple
        :return: Tuple
        """
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
        """
        Return current action
        :return: Tuple
        """
        return self.current_action

    def get_all_actions(self):
        """
        Returns action space
        :return: Set
        """
        return self.possible_actions
