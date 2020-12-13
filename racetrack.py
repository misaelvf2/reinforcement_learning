import numpy as np
from copy import deepcopy


# Velocity limits
X_VEL_LO_LIM = -5
X_VEL_UP_LIM = 5
Y_VEL_LO_LIM = -5
Y_VEL_UP_LIM = 5

# State transition probabilities
ACTION_SUCCESS_PROB = 0.8
ACTION_FAIL_PROB = 0.2

class RaceTrack:
    """
    Class for representing and abstracting the RaceTrack environment
    """
    def __init__(self, num_rows, num_cols, layout, initial_state, reset_on_crash=False):
        """
        Initializes class
        :param num_rows: Int
        :param num_cols: Int
        :param layout: List
        :param initial_state: Tuple
        :param reset_on_crash: Boolean
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.layout = layout
        self.initial_state = initial_state
        self.current_state = deepcopy(self.initial_state)
        self.reset_on_crash = reset_on_crash

    def next_states(self, action):
        """
        Returns possible next states with corresponding probabilities
        :param action: Tuple
        :return: List
        """
        # Apply accelerations and update positions
        new_x_velocity = min(max(self.current_state[2] + action[0], X_VEL_LO_LIM), X_VEL_UP_LIM)
        new_y_velocity = min(max(self.current_state[3] + action[1], Y_VEL_LO_LIM), Y_VEL_UP_LIM)
        new_x_position = min(max(self.current_state[0] + new_x_velocity, 0), self.num_cols)
        new_y_position = min(max(self.current_state[1] + new_y_velocity, 0), self.num_rows)

        unexpected_x_velocity = min(max(self.current_state[2], X_VEL_LO_LIM), X_VEL_UP_LIM)
        unexpected_y_velocity = min(max(self.current_state[3], Y_VEL_LO_LIM), Y_VEL_UP_LIM)
        unexpected_x_position = min(max(self.current_state[0] + unexpected_x_velocity, 0), self.num_cols)
        unexpected_y_position = min(max(self.current_state[1] + unexpected_y_velocity, 0), self.num_rows)

        # Create and validate new state
        new_state = (new_x_position, new_y_position, new_x_velocity, new_y_velocity)
        new_state = self.__validate_state(new_state)
        unexpected_new_state = (unexpected_x_position, unexpected_y_position, unexpected_x_velocity, unexpected_y_velocity)
        unexpected_new_state = self.__validate_state(unexpected_new_state)
        possible_states = [(new_state, ACTION_SUCCESS_PROB), (unexpected_new_state, ACTION_FAIL_PROB)]
        return possible_states

    def update_state(self, action, indicate_random=False):
        """
        Updates state stochastically
        :param action: Tuple
        :param indicate_random: Boolean
        :return: Tuple
        """
        # Get possible next states, choose one probabilistically, and update current state
        next_states = self.next_states(action)
        state_map = {0: next_states[0][0], 1: next_states[1][0]}
        next_state_choice = np.random.choice([0, 1], p=[next_states[0][1], next_states[1][1]])
        if next_state_choice == 1 and indicate_random:
            print("Nondeterministic response")
        self.current_state = state_map[next_state_choice]
        return self.current_state

    def set_state(self, state):
        """
        Sets current state
        :param state: Tuple
        :return: None
        """
        self.current_state = state

    def get_state(self):
        """
        Returns current state
        :return: Tuple
        """
        return self.current_state

    def reset_state(self):
        """
        Resets current state to initial state
        :return: Tuple
        """
        self.current_state = self.initial_state
        return self.current_state

    def in_terminal_state(self):
        """
        Indicates that environment is in terminal state
        :return: Boolean
        """
        return self.layout[self.current_state[1]][self.current_state[0]] == 'F'

    def __validate_state(self, state):
        """
        Validates and corrects given state using Bresenham's algorithm to detect collision
        :param state: Tuple
        :return: Tuple
        """
        line = list(self.bresenham(self.current_state[0], self.current_state[1], state[0], state[1]))
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
            if self.reset_on_crash:
                x_position, y_position, x_velocity, y_velocity = self.initial_state
            else:
                x_velocity = 0
                y_velocity = 0
        new_state = (x_position, y_position, x_velocity, y_velocity)
        return new_state

    def get_all_states(self):
        """
        Returns state space
        :return: Set
        """
        states = {(x, y, vx, vy) for x in range(0, self.num_cols) for y in range(0, self.num_rows) for vx in range(X_VEL_LO_LIM, X_VEL_UP_LIM + 1) for vy in
                  range(Y_VEL_LO_LIM, Y_VEL_UP_LIM + 1) if self.layout[y][x] != '#'}
        return states

    def get_next_states(self, state, action):
        """
        Returns possible next states with corresponding probabilities
        :param state: Tuple
        :param action: Tuple
        :return: List
        """
        # Apply accelerations and update positions
        new_x_velocity = min(max(state[2] + action[0], X_VEL_LO_LIM),
                             X_VEL_UP_LIM)
        new_y_velocity = min(max(state[3] + action[1], Y_VEL_LO_LIM),
                             Y_VEL_UP_LIM)
        new_x_position = min(max(state[0] + new_x_velocity, 0), self.num_cols)
        new_y_position = min(max(state[1] + new_y_velocity, 0), self.num_rows)

        unexpected_x_velocity = min(max(self.current_state[2], X_VEL_LO_LIM), X_VEL_UP_LIM)
        unexpected_y_velocity = min(max(self.current_state[3], Y_VEL_LO_LIM), Y_VEL_UP_LIM)
        unexpected_x_position = min(max(self.current_state[0] + unexpected_x_velocity, 0), self.num_cols)
        unexpected_y_position = min(max(self.current_state[1] + unexpected_y_velocity, 0), self.num_rows)

        # Create and validate new state
        new_state = (new_x_position, new_y_position, new_x_velocity, new_y_velocity)
        new_state = self.__validate_state(new_state)
        unexpected_new_state = (unexpected_x_position, unexpected_y_position, unexpected_x_velocity, unexpected_y_velocity)
        unexpected_new_state = self.__validate_state(unexpected_new_state)
        possible_states = [(new_state, ACTION_SUCCESS_PROB), (unexpected_new_state, ACTION_FAIL_PROB)]
        return possible_states

    def get_reward(self, state):
        """
        Returns reward for being in given state
        :param state: Tuple
        :return: Int
        """
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
            print("Invalid state")

    def reward(self):
        """
        Returns reward for being in current state
        :return: Int
        """
        x_position = self.current_state[0]
        y_position = self.current_state[1]
        track_symbol = self.layout[y_position][x_position]
        if  track_symbol == '.':
            return 1
        elif track_symbol == 'S':
            return 1
        elif track_symbol == 'F':
            return 0

    def bresenham(self, x0, y0, x1, y1):
        """
        Implements Bresenham's algorithm for generating line from
        starting position, to ending position
        :param y0: Int
        :param x1: Int
        :param y1: Int
        :return: Generator
        """
        x_change = x1 - x0
        y_change = y1 - y0
        x_sign = 1 if x_change > 0 else -1
        y_sign = 1 if y_change > 0 else -1
        x_change = abs(x_change)
        y_change = abs(y_change)

        if x_change > y_change:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            x_change, y_change = y_change, x_change
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * y_change - x_change
        y = 0
        for x in range(x_change + 1):
            yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
            if D >= 0:
                y += 1
                D -= 2 * x_change
            D += 2 * y_change
