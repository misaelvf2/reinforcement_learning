from car import Car
from racetrack import RaceTrack
from value_iteration_1 import ValueIteration
from q_learning_1 import QLearning


def main_value_iteration():
    with open('L-track.txt') as f:
        specs = f.readline().strip().split(',')
        rows = int(specs[0])
        cols = int(specs[1])
        layout = f.read().splitlines()

        initial_state = (1, 8, 0, 0)
        initial_action = (0, 0)

        agent = Car(initial_action)
        environment = RaceTrack(rows, cols, layout, initial_state)

        discount = 0.9
        threshold = 0.0001
        max_iterations = 1000
        value_iterator = ValueIteration(discount, threshold, max_iterations, environment, agent)
        value_iterator.run()
        path = value_iterator.extract_policy(initial_state)
        draw_track(path, layout)

def draw_track(path, track):
    print("Cost: ", len(path))
    for step in path:
        old_line = track[step[1]]
        new_line = old_line[:step[0]] + 'C' + old_line[step[0] + 1:]
        track[step[1]] = new_line
    for line in track:
        print(line)

def main():
    with open('L-track.txt') as f:
        specs = f.readline().strip().split(',')
        rows = int(specs[0])
        cols = int(specs[1])
        layout = f.read().splitlines()

        initial_state = (1, 8, 0, 0)
        initial_action = (0, 0)
        epsilon = 0.5

        agent = Car(initial_action, epsilon)
        environment = RaceTrack(rows, cols, layout, initial_state)

        discount = 0.9
        learning_rate = 0.9
        threshold = 10
        max_iterations = 10000
        q_learner = QLearning(discount, learning_rate, threshold, max_iterations, environment, agent)
        path = q_learner.run()
        draw_track(path, layout)

main_value_iteration()
