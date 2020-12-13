import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class ValueIteration:
    """
    Class that implements the Value Iteration algorithm
    """
    def __init__(self, discount, threshold, max_iterations, environment, agent):
        """
        Initializes algorithm
        :param discount: Float
        :param threshold: Float
        :param max_iterations: Int
        :param environment: RaceTrack
        :param agent: Car
        """
        self.discount = discount
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.environment = environment
        self.agent = agent
        self.max_diffs = []

    def __initialize_values(self):
        """
        Initializes value function
        :return: None
        """
        self.all_states = self.environment.get_all_states()
        self.all_actions = self.agent.get_all_actions()
        self.value_function = {k: v for (k, v) in zip(self.all_states, np.zeros(len(self.all_states)))}

    def run(self):
        """
        Runs algorithm
        :return: Dict
        """
        self.__initialize_values()
        differences = [np.inf for _ in range(len(self.value_function))]
        i = 0
        while i < self.max_iterations and max(differences) > self.threshold:
            new_value_function = deepcopy(self.value_function)
            for state in self.all_states:
                self.environment.set_state(state)
                state_action_pairs = [(state, action) for _ in range(1) for action in self.all_actions]
                state_action_function = {k: v for (k, v) in zip(state_action_pairs, np.zeros(len(self.all_actions)))}
                for action in self.all_actions:
                    state_action_function[(state, action)] = self.environment.reward()
                    next_possible_states = self.environment.next_states(action)
                    for possible_state in next_possible_states:
                        next_state = possible_state[0]
                        probability = possible_state[1]
                        state_action_function[(state, action)] += self.discount * probability * self.value_function[next_state]
                new_value_function[state] = min(state_action_function.values())
            differences = np.array(list(new_value_function.values()) - np.array(list(self.value_function.values())))
            print("Maximum difference: ", max(differences))
            self.max_diffs.append(max(differences))
            self.value_function = new_value_function
            i += 1
            print("Iterations: ", i)
        return self.value_function

    def extract_policy(self, state):
        """
        Extracts and follows policy using trained value function
        :param state: Tuple
        :return: List
        """
        self.environment.set_state(state)
        current_state = state
        path = [current_state]
        track = self.environment.layout
        while track[current_state[1]][current_state[0]] != 'F':
            next_action = (0, 0)
            orig_state = current_state
            optimal_state, optimal_value = current_state, np.inf
            for action in self.all_actions:
                self.environment.update_state(action)
                new_state = self.environment.get_state()
                action_value = self.value_function[new_state]
                if action_value < optimal_value:
                    optimal_state = new_state
                    optimal_value = action_value
                    next_action = action
                self.environment.set_state(current_state)
            current_state = optimal_state
            current_value = optimal_value
            current_state = self.environment.update_state(next_action, indicate_random=True)
            # print(current_state, current_value, track[current_state[1]][current_state[0]])
            print(orig_state, current_state, next_action, '\n')
            path.append(current_state)
        print("Finished training---------------------------------------")
        return path

    def plot_max_diffs(self):
        """
        Plot max difference over time
        :return: None
        """
        plt.plot(self.max_diffs)
        plt.ylabel('Maximum Difference')
        plt.savefig("max_diffs_graph.png")
