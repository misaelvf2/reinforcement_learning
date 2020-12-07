from bresenham import bresenham


class Car:
    def __init__(self, position, velocity, acceleration, track):
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.track = track

    def accelerate(self, action):
        self.acceleration['x_acceleration'] = action['x_action']
        self.acceleration['y_acceleration'] = action['y_action']
        self.__update_velocity()
        self.__update_position()
        self.__update_track()

    def __update_velocity(self):
        self.velocity['x_velocity'] += self.acceleration['x_acceleration']
        self.velocity['y_velocity'] += self.acceleration['y_acceleration']
        if self.velocity['x_velocity'] <= 0:
            self.velocity['x_velocity'] = max(-5, self.velocity['x_velocity'])
        else:
            self.velocity['x_velocity'] = min(5, self.velocity['x_velocity'])
        if self.velocity['y_velocity'] <= 0:
            self.velocity['y_velocity'] = max(-5, self.velocity['y_velocity'])
        else:
            self.velocity['y_velocity'] = min(5, self.velocity['y_velocity'])

    def __update_position(self):
        new_position = {'x_coordinate': self.position['x_coordinate'] + self.velocity['x_velocity'],
                        'y_coordinate': self.position['y_coordinate'] + self.velocity['y_velocity']}
        self.position = self.__detect_collision(self.position, new_position)

    def __detect_collision(self, old_position, new_position):
        line = list(bresenham(old_position['x_coordinate'], old_position['y_coordinate'], new_position['x_coordinate'],
                         new_position['y_coordinate']))
        i = 0
        new_position['x_coordinate'] = line[i][0]
        new_position['y_coordinate'] = line[i][1]
        collision = False
        while i < len(line) and not collision:
            if self.track[line[i][1]][line[i][0]] != '#':
                new_position['x_coordinate'] = line[i][0]
                new_position['y_coordinate'] = line[i][1]
            else:
                collision = True
            i += 1
        if collision:
            self.velocity['x_velocity'] = 0
            self.velocity['y_velocity'] = 0
        return new_position

    def __update_track(self):
        old_line = self.track[self.position['y_coordinate']]
        new_line = old_line[:self.position['x_coordinate']] + 'C' + old_line[self.position['x_coordinate'] + 1:]
        self.track[self.position['y_coordinate']] = new_line

    def print_track(self):
        for line in self.track:
            print(line)

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration