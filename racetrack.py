import numpy as np
from copy import deepcopy
from bresenham import bresenham

# Velocity limits
X_VEL_LO_LIM = -5
X_VEL_UP_LIM = 5
Y_VEL_LO_LIM = -5
Y_VEL_UP_LIM = 5

# State transition probabilities
ACTION_SUCCESS_PROB = 0.8
ACCESS_FAIL_PROB = 0.2

class RaceTrack:
    def __init__(self, num_rows, num_cols, layout, initial_state):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.layout = layout
        self.initial_state = initial_state
        self.current_state = deepcopy(self.initial_state)

    def next_states(self, action):
        # Apply accelerations and update positions
        new_x_velocity = min(max(self.current_state[2] + action[0], X_VEL_LO_LIM),
                             X_VEL_UP_LIM)
        new_y_velocity = min(max(self.current_state[3] + action[1], Y_VEL_LO_LIM),
                             Y_VEL_UP_LIM)
        new_x_position = min(max(self.current_state[0] + new_x_velocity, 0), self.num_cols)
        new_y_position = min(max(self.current_state[1] + new_y_velocity, 0), self.num_rows)

        # Create and validate new state
        new_state = (new_x_position, new_y_position, new_x_velocity, new_y_velocity)
        new_state = self.__validate_state(new_state)
        possible_states = [(new_state, ACTION_SUCCESS_PROB), (self.current_state, ACCESS_FAIL_PROB)]
        return possible_states

    def update_state(self, action):
        # Get possible next states, choose one probabilistically, and update current state
        next_states = self.next_states(action)
        state_map = {0: next_states[0][0], 1: next_states[1][0]}
        next_state_choice = np.random.choice([0, 1], p=[next_states[0][1], next_states[1][1]])
        self.current_state = state_map[next_state_choice]
        return self.current_state

    def set_state(self, state):
        self.current_state = state

    def get_state(self):
        return self.current_state

    def reset_state(self):
        self.current_state = self.initial_state
        return self.current_state

    def in_terminal_state(self):
        return self.layout[self.current_state[1]][self.current_state[0]] == 'F'

    def __validate_state(self, state):
        line = list(bresenham(self.current_state[0], self.current_state[1], state[0], state[1]))
        i = 0
        collision = False
        x_position = state[0]
        y_position = state[1]
        x_velocity = state[2]
        y_velocity = state[3]
        while i < len(line) and not collision:
            if self.layout[line[i][1]][line[i][0]] != '#':
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

    def get_all_states(self):
        states = {(x, y, vx, vy) for x in range(0, self.num_cols) for y in range(0, self.num_rows) for vx in range(X_VEL_LO_LIM, X_VEL_UP_LIM + 1) for vy in
                  range(Y_VEL_LO_LIM, Y_VEL_UP_LIM + 1) if self.layout[y][x] != '#'}
        return states

    def get_next_states(self, state, action):
        # Apply accelerations and update positions
        new_x_velocity = min(max(state[2] + action[0], X_VEL_LO_LIM),
                             X_VEL_UP_LIM)
        new_y_velocity = min(max(state[3] + action[1], Y_VEL_LO_LIM),
                             Y_VEL_UP_LIM)
        new_x_position = min(max(state[0] + new_x_velocity, 0), self.num_cols)
        new_y_position = min(max(state[1] + new_y_velocity, 0), self.num_rows)

        # Create and validate new state
        new_state = (new_x_position, new_y_position, new_x_velocity, new_y_velocity)
        new_state = self.__validate_state(new_state)
        possible_states = [(new_state, ACTION_SUCCESS_PROB), (self.current_state, ACCESS_FAIL_PROB)]
        return possible_states

    def get_reward(self, state):
        x_position = state[0]
        y_position = state[1]
        track_symbol = self.layout[y_position][x_position]
        if  track_symbol == '.':
            return 1
        elif track_symbol == 'S':
            return 1
        elif track_symbol == 'F':
            return 0
        else:
            print(state)

    def reward(self):
        x_position = self.current_state[0]
        y_position = self.current_state[1]
        track_symbol = self.layout[y_position][x_position]
        if  track_symbol == '.':
            return 1
        elif track_symbol == 'S':
            return 1
        elif track_symbol == 'F':
            return 0
