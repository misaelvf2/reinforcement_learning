import numpy as np
from copy import deepcopy
import math
from bresenham import bresenham


class QLearning:
    def __init__(self, states, actions, discount, learning_rate, epsilon, track):
        self.states = states
        self.actions = actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.track = track
        self.q_function = {}

    # def initialize_q(self):
    #     q_function = {((0, 0, 0, 0), (0, 0)): 0}
    #     state_action_pair = ((0, 0, 0, 0), (0, 0))
    #     if state_action_pair in q_function:
    #         print("Yes")

    def run(self):
        i = 0
        path_costs = []
        cost = math.inf
        while i < 1000:
            # print("Length: ", len(self.q_function.keys()))
            initial_state = (2, 10, 0, 0)
            current_state = initial_state
            path = []
            while self.track[current_state[1]][current_state[0]] != 'F':
                path.append(current_state)
                current_action = self.__choose_action(current_state)
                next_state = self.__next_states(current_state, current_action)
                reward = self.__get_reward(current_state)
                state_action_pair = (current_state, current_action)
                next_action = self.__choose_action(next_state)
                next_state_action_pair = (next_state, next_action)
                if next_state_action_pair in self.q_function:
                    next_q_value = self.q_function[next_state_action_pair]
                else:
                    self.q_function[next_state_action_pair] = np.random.randint(3, 10) / 1.0
                    next_q_value = np.random.randint(3, 10) / 1.0
                if state_action_pair not in self.q_function:
                    rand_init = np.random.randint(3, 6) / 1.0
                    self.q_function[state_action_pair] = rand_init + self.learning_rate * (reward + self.discount * next_q_value - rand_init)
                else:
                    self.q_function[state_action_pair] += self.learning_rate * (reward + self.discount * next_q_value - self.q_function[state_action_pair])
                current_state = next_state
                # print(current_state)
            path_costs.append(self.draw_track(path))
            # print("Terminal state!")
            i += 1
            self.learning_rate -= 0.001 # 0.1
            self.learning_rate = max(self.learning_rate, 0.05) # 0.05
            self.epsilon -= 0.001 # 0.1
            self.epsilon = max(self.epsilon, 0.05) # 0.05
            cost = path_costs[-1]
        print(path_costs)


    def __next_states(self, state, action):
        x_velocity = min(max(state[2] + action[0], -5), 5)
        y_velocity = min(max(state[3] + action[1], -5), 5)
        x_position = min(max(state[0] + x_velocity, 0),  24)
        y_position = min(max(state[1] + y_velocity, 0), 24)

        line = list(bresenham(state[0], state[1], x_position, y_position))
        i = 0
        x_position = line[i][0]
        y_position = line[i][1]
        collision = False
        while i < len(line) and not collision:
            if self.track[line[i][1]][line[i][0]] != '#':
                x_position = line[i][0]
                y_position = line[i][1]
            else:
                collision = True
            i += 1
        if collision:
            x_velocity = 0
            y_velocity = 0
        new_state = (x_position, y_position, x_velocity, y_velocity)
        state_map = {
            0: new_state,
            1: state
        }
        actual_state = np.random.choice([0, 1], p=[0.8, 0.2])
        return state_map[actual_state]

    def __get_reward(self, state):
        x_coordinate = state[0]
        y_coordinate = state[1]
        track_symbol = self.track[y_coordinate][x_coordinate]
        if  track_symbol == '.':
            return 1
        elif track_symbol == 'S':
            return 1
        elif track_symbol == 'F':
            return 0

    def __choose_action(self, state):
        actions = {(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)}
        optimal_state, optimal_value = state, math.inf
        optimal_action = (0, 0)
        for action in actions:
            x_velocity = min(max(state[2] + action[0], -5), 5)
            y_velocity = min(max(state[3] + action[1], -5), 5)
            x_position = min(max(state[0] + x_velocity, 0), 24)
            y_position = min(max(state[1] + y_velocity, 0), 24)
            new_state = (x_position, y_position, x_velocity, y_velocity)
            new_state = self.__validate_state(state, new_state, self.track)
            state_action_pair = (new_state, action)
            if state_action_pair in self.q_function:
                state_action_pair_value = self.q_function[state_action_pair]
            else:
                self.q_function[state_action_pair] = np.random.randint(0, 10) / 1.0
                state_action_pair_value = np.random.randint(0, 10) / 1.0
            if state_action_pair_value < optimal_value:
                optimal_state = new_state
                optimal_action = action
                optimal_value = state_action_pair_value
        choice_map = {
            0: optimal_action,
            1: (-1, -1),
            2: (0, -1),
            3: (0, 1),
            4: (1, 0),
            5: (1, 1),
            6: (-1, 0)
        }
        prob = self.epsilon/6.0
        taken_action = np.random.choice([0, 1, 2, 3, 4, 5, 6],
                                        p=[1 - self.epsilon, prob, prob, prob, prob, prob, prob])
        return choice_map[taken_action]

    def __validate_state(self, old_state, new_state, track):
        line = list(bresenham(old_state[0], old_state[1], new_state[0], new_state[1]))
        i = 0
        x_position = line[i][0]
        y_position = line[i][1]
        x_velocity = new_state[2]
        y_velocity = new_state[3]
        collision = False
        while i < len(line) and not collision:
            if track[line[i][1]][line[i][0]] != '#':
                x_position = line[i][0]
                y_position = line[i][1]
            else:
                collision = True
            i += 1
        if collision:
            x_velocity = 0
            y_velocity = 0
        new_state = (x_position, y_position, x_velocity, y_velocity)
        return new_state

    def draw_track(self, path):
        new_track = self.track.copy()
        for step in path:
            old_line = new_track[step[1]]
            new_line = old_line[:step[0]] + 'C' + old_line[step[0] + 1:]
            new_track[step[1]] = new_line
        # for line in new_track:
        #     print(line)
        # print(len(path), self.epsilon, self.learning_rate)
        print(len(path))
        return len(path)
