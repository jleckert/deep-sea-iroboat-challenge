import numpy as np
from astropy.coordinates import spherical_to_cartesian
from ai import cs


class Sphere:

    def __init__(self, radius):
        """
        Define a sphere object.

        :param radius: radius of the sphere
        """
        self.r = radius
        self.c = 2 * np.pi * self.r

    def rotate(self, x, lon, lat):
        """
        Rotate a set of vectors to be tangent to the sphere
        at latitude lat and longitude lon.

        :param x: vectors to be rotated
        :param lon: longitude
        :param lat: latitude
        :return: x rotated to be tangent to the sphere at
            (lat, lon)
        """
        lon = np.radians(lon)
        lat = np.radians(-lat)
        return self._rotation_z(self._rotation_y(x, lat), lon)

    def _rotation_x(self, x, theta):
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        return rot.dot(x).astype(np.float)

    def _rotation_y(self, x, theta):
        rot = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        return rot.dot(x).astype(np.float)

    def _rotation_z(self, x, theta):
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return rot.dot(x).astype(np.float)

    def move3(self, origin, d):
        """
        Move from the origin point displaced by displacement
        vectors d, account for the curvature of the sphere.

        :param origin: point to move from
        :param d: displacement vectors
        :return: new points after displacement
        """

        d *= 111.19492664

        origin_ = spherical_to_cartesian(self.r, np.radians(origin[0]), np.radians(origin[1]))
        origin_ = np.array(origin_)

        d = self.rotate(d, origin[1], origin[0])

        speed = np.sqrt(np.sum(d**2, axis=0))

        angle = (speed/self.c) * 2 * np.pi

        speed = np.tan(angle) * self.r

        vel = np.linalg.norm(d, axis=0)
        vel[vel == 0] = 1
        vel = (d/vel) * speed

        result = origin_ + vel.T

        result_ = result.T
        result = cs.cart2sp(x=result_[0], y=result_[1], z=result_[2])

        result = np.degrees(np.array([result[1], result[2]]))

        return result

