import numpy as np


class WindEnvelope:

    def __init__(self, wind_data, sphere):
        """
        A class that defines methods to obtain the displacement
        envelope for boats relative to wind speed and direction
        vectors.

        :param wind_data: table of velocity values relative to
            wind direction and speed
        :param sphere: sphere object that handles projection
            operations
        """
        self.data = wind_data
        self.data[self.data < 0.001] = 0.001
        self.data *= 1.852
        self.data *= 360/40075
        self.data = np.vstack((self.data, np.flip(self.data, axis=0)[1:-1]))
        self.sphere = sphere

    def compute(self, wind_u, wind_v, x, y, theta, scale=1.):
        """
        Compute displacement relative to wind direction vectors
        and positions.

        :param wind_u: wind vector u-components
        :param wind_v: wind vector v-components
        :param x: longitudal positions
        :param y: latitudal positions
        :param theta: Angular resolution to compute paths for
        :param scale: how much to scale the displacement by based
            on the hour delta
        :return: new displaced positions
        """
        try:
            _ = iter(x)
        except TypeError:
            x, y = np.array([x]), np.array([y])

        origin = np.array([x, y])

        _ = np.array([-1, 0])

        mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        surround = []
        angles = []

        ctr = 0
        while ctr < np.pi*2:
            angles.append(np.round(ctr*180/np.pi))
            surround.append(np.copy(_))
            _ = mat.dot(_)
            ctr += theta

        surround = np.array(surround)
        angles = np.array(angles, dtype=np.int32)

        ret = []

        try:
            _ = iter(wind_u)
        except TypeError:
            wind_u, wind_v = [wind_u], [wind_v]

        for _dx, _dy in zip(wind_u, wind_v):

            _ = np.array([_dx, _dy])
            s = np.sqrt(np.sum(_ ** 2))

            s *= 70
            s = int(max(0, min(69, s)))

            _ /= np.sqrt(np.sum(_**2))

            target = np.array([1, 0])

            a = np.math.atan2(np.linalg.det([_, target]), np.dot(_, target))

            t_angle = int(np.degrees(a))
            t_angle = t_angle + 360 if t_angle < 0 else t_angle
            t_angle = 360 - t_angle

            wind_strengths = self.data[:, s]

            wind_strengths = np.concatenate(
                (wind_strengths[-t_angle:], wind_strengths[:-t_angle]))
            wind_strengths = wind_strengths[angles]

            r = (surround.T * wind_strengths).T
            r *= scale

            ret.append(r)

        _ret = []

        for r, o in zip(ret, origin.T):

            r = r.T

            r = np.vstack((np.zeros(len(ret[0])), r))
            r = self.sphere.move3(np.array([o[1], o[0]]), r)
            r[r > 180] -= 360
            r = np.flip(r, axis=0)

            _ret.append(r.T)

        ret = np.array(_ret)

        return ret
