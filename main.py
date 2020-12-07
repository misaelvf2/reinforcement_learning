from car import Car
from value_iteration import ValueIteration
import math
from bresenham import bresenham


def main_1():
    with open('O-track.txt') as f:
        specs = f.readline().strip().split(',')
        track = f.read().splitlines()
        possible_actions = {-1, 0, 1}

        position = {'x_coordinate': 2, 'y_coordinate': 10}
        velocity = {'x_velocity': 0, 'y_velocity': 0}
        acceleration = {'x_acceleration': 0, 'y_acceleration': 0}

        my_car = Car(position, velocity, acceleration, track)
        action = {'x_action': 0, 'y_action': -1}
        for i in range(10):
            my_car.accelerate(action)
        my_car.print_track()

def validate_state(old_state, new_state, track):
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

def main():
    with open('O-track.txt') as f:
        specs = f.readline().strip().split(',')
        rows = int(specs[0])
        cols = int(specs[1])
        track = f.read().splitlines()
        actions = {(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)}

        states = {(x, y, vx, vy) for x in range(0, rows) for y in range(0, cols) for vx in range(-5, 6) for vy in range(-5, 6) if track[y][x] != '#'}
        discount = 0.9
        threshold = 0.01
        max_iterations = 1000
        value_iterator = ValueIteration(states, actions, discount, threshold, max_iterations, track)
        values = value_iterator.run()
        path = extract_policy((2, 10, 0, 0), values, track, rows, cols)
        draw_track(path, track)

def draw_track(path, track):
    for step in path:
        old_line = track[step[1]]
        new_line = old_line[:step[0]] + 'C' + old_line[step[0] + 1:]
        track[step[1]] = new_line
    for line in track:
        print(line)


def extract_policy(state, values, track, rows, cols):
    actions = {(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)}
    value = values[state]
    path = []
    while track[state[1]][state[0]] != 'F':
        optimal_state, optimal_value = state, math.inf
        for action in actions:
            x_velocity = min(max(state[2] + action[0], -5), 5)
            y_velocity = min(max(state[3] + action[1], -5), 5)
            x_position = min(max(state[0] + x_velocity, 0), rows - 1)
            y_position = min(max(state[1] + y_velocity, 0), cols - 1)
            new_state = (x_position, y_position, x_velocity, y_velocity)
            new_state = validate_state(state, new_state, track)
            action_value = values[new_state]
            if action_value < optimal_value:
                optimal_state = new_state
                optimal_value = action_value
        value = optimal_value
        state = optimal_state
        print(state, value, track[state[1]][state[0]])
        path.append(state)
    return path

if __name__ == '__main__':
    main()
