import numpy as np
from copy import deepcopy
from bresenham import bresenham


class ValueIteration:
    def __init__(self, states, actions, discount, threshold, max_iterations, track):
        self.states = states
        self.actions = actions
        self.discount = discount
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.track = track

    def __initialize_values(self):
        self.values = {k: v for (k, v) in zip(self.states, np.zeros(len(self.states)))}
        # self.values = {k: v for (k, v) in zip(self.states, [30 for _ in range(len(self.states))])}

    def run(self):
        self.__initialize_values()
        i = 0
        differences = [np.inf for _ in range(len(self.values))]
        while i < self.max_iterations and max(differences) > self.threshold:
            new_values = deepcopy(self.values)
            for state in self.states:
                states_actions = [(state, action) for s in range(1) for action in self.actions]
                states_actions_values = {k: v for (k, v) in zip(states_actions, np.zeros(len(self.actions)))}
                for action in self.actions:
                    states_actions_values[(state, action)] = self.__get_reward(state)
                    next_states = self.__next_states(state, action)
                    for next_state in next_states:
                        states_actions_values[(state, action)] += self.discount * next_state[1] * self.values[next_state[0]]
                new_values[state] = min(states_actions_values.values())
            differences = np.array(list(new_values.values())) - np.array(list(self.values.values()))
            print(max(differences))
            self.values = new_values
            i += 1
        return self.values

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
        return [(new_state, 0.8), (state, 0.2)]


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
