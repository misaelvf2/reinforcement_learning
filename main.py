from car import Car
from racetrack import RaceTrack
from value_iteration import ValueIteration
from q_learning import QLearning
from sarsa import Sarsa


def draw_track(path, track):
    """
    Draws track from given path
    :param path: List
    :param track: List
    :return: None
    """
    print("Cost: ", len(path))
    for step in path:
        old_line = track[step[1]]
        new_line = old_line[:step[0]] + 'C' + old_line[step[0] + 1:]
        track[step[1]] = new_line
    for line in track:
        print(line)

def main(algorithm, track, x_start, y_start, discount, learning_rate, threshold, max_iterations, epsilon=None, reset_on_crash=False):
    """
    Program entry. Runs selected algorithm on selected track, at given coordinates, with given parameters
    :param algorithm: String
    :param track: List
    :param x_start: Int
    :param y_start: Int
    :param discount: Float
    :param learning_rate: Float
    :param threshold: Float
    :param max_iterations: Int
    :param epsilon: Float
    :param reset_on_crash: Boolean
    :return: None
    """
    with open(track) as f:
        specs = f.readline().strip().split(',')
        rows = int(specs[0])
        cols = int(specs[1])
        layout = f.read().splitlines()

        initial_state = (x_start, y_start, 0, 0)
        initial_action = (0, 0)

        agent = Car(initial_action, epsilon)
        environment = RaceTrack(rows, cols, layout, initial_state, reset_on_crash=reset_on_crash)

        if algorithm == 'value_iteration':
            value_iterator = ValueIteration(discount, threshold, max_iterations, environment, agent)
            value_iterator.run()
            path = value_iterator.extract_policy(initial_state)
            value_iterator.plot_max_diffs()
        elif algorithm == 'q_learning':
            q_learner = QLearning(discount, learning_rate, threshold, max_iterations, environment, agent)
            path = q_learner.run()
            q_learner.plot_avg_cost()
        elif algorithm == 'sarsa':
            sarsa = Sarsa(discount, learning_rate, threshold, max_iterations, environment, agent)
            path = sarsa.run()
            sarsa.plot_avg_cost()
        else:
            print("No algorithm selected")
            return None
        draw_track(path, layout)


# O-track: (2, 10)
# L-track: (1, 8)
# R-track: (2, 26)
main('value_iteration', 'R-track.txt', x_start=2, y_start=26, discount=0.9, learning_rate=0.9, threshold=0.0001, max_iterations=10000, reset_on_crash=False)
# main('sarsa', 'L-track.txt', x_start=1, y_start=8, discount=0.9, learning_rate=0.9, threshold=30, max_iterations=10000, epsilon=0.5, reset_on_crash=False)
# main('q_learning', 'O-track.txt', x_start=2, y_start=10, discount=0.9, learning_rate=0.9, threshold=30, max_iterations=10000, epsilon=0.5, reset_on_crash=False)