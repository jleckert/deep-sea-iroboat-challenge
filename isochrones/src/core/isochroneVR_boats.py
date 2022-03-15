import numpy as np


class BaseBoat:

    def __init__(self, x, y, team, team_color, bounds):
        """
        An example boat class to test out different strategies.

        :param x:
        :param y:
        :param team:
        :param team_color:
        :param bounds:
        :param envelope:
        """
        # self.envelope = envelope

        self.pos = np.array([x, y])
        self.max_speed = 0.1
        self.finished = False
        self.team = team
        self.teamcolor = team_color
        self.time = 0

        self.x1, self.x2, self.y1, self.y2 = bounds

        self.pos_s = []

    def step(self, dx, dy, dx2, dy2):
        """
        Use this to move a boat into a certain direction, scaled
        by the boat's maximum speed.

        :param dx:
        :param dy:
        :param dx2:
        :param dy2:
        :return:
        """
        self.pos_s.append(np.copy(self.pos))
        if self.finished:
            return True

        d = np.array([dx, dy]).astype(np.float32)
        d /= np.sqrt(np.sum(d**2))
        d *= self.max_speed

        self.pos += d
        self.pos += np.array([dx2, dy2])

        self.pos[0] = np.clip(self.pos[0], self.x1, self.x2)
        self.pos[1] = np.clip(self.pos[1], self.y1, self.y2)

        if self.pos[1] >= self.y2:
            return True

        return False
