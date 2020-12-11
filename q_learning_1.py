import numpy as np
from collections import deque


class QLearning:
    def __init__(self, discount, learning_rate, threshold, max_iterations, environment, agent):
        self.discount = discount
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.environment = environment
        self.agent = agent

    def __initialize_values(self):
        self.all_states = self.environment.get_all_states()
        self.all_actions = self.agent.get_all_actions()
        self.state_action_pairs = {(s, a) for s in self.all_states for a in self.all_actions}
        self.q_function = {k: v for (k, v) in zip(self.state_action_pairs, np.zeros(len(self.state_action_pairs)))}
        self.agent.set_q_function(self.q_function)

    def run(self):
        self.__initialize_values()
        q = deque(maxlen=25)
        cost = np.inf
        i = 0
        while cost > self.threshold and i < self.max_iterations:
            initial_state = self.environment.reset_state()
            current_state = initial_state
            path = [current_state]
            while not self.environment.in_terminal_state():
                current_action = self.agent.take_action(current_state)
                next_state = self.environment.update_state(current_action)
                reward = self.environment.get_reward(next_state)
                next_action = self.agent.take_action(next_state)
                current_q = self.q_function[(current_state, current_action)]
                next_q = self.q_function[(next_state, next_action)]
                try:
                    self.q_function[(current_state, current_action)] += self.learning_rate * (reward + self.discount * next_q - current_q)
                except TypeError:
                    print(reward, self.discount)
                current_state = next_state
                path.append(current_state)
            self.learning_rate -= 0.0001
            self.learning_rate = max(self.learning_rate, 0.001)
            q.appendleft(len(path))
            if len(q) == 25:
                cost = sum(q) / len(q)
                print(cost)
                q.pop()
            i += 1
            # print(len(path))
        # print(len(path))
        return path
