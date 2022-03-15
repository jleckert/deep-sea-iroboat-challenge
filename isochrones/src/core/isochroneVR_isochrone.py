import os
import numpy as np
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.errors import TopologicalError
import random
import matplotlib.pyplot as plt
import time
from shapely.ops import cascaded_union

import geopandas
from matplotlib import path
from shapely.geometry import LineString, MultiLineString
from src.core.bresenham_np import bresenhamline

from multiprocessing.dummy import Pool as ThreadPool


from src.common.log import logger


def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    ret = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    return ret


def chunks(a, n):
    """
    Divides the list a into n approximately even smaller
    lists.

    :param a: List to be divided.
    :param n: Number of chunks to divide the list into.
    :return: List of n sub-lists.
    """

    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def path_array(ps, w, h, bounds):
    """
    Converts a list of paths ps into a collision array.

    :param ps: Paths list.
    :param w: Width of the array.
    :param h: Height of the array
    :param bounds: Bounds of the map coordinates
        (typically -180, 180, -90, 90)
    :return: Collision numpy array.
    """
    X, Y = np.meshgrid(np.linspace(
        bounds[0], bounds[1], w), np.linspace(bounds[2], bounds[3], h))
    flags = np.zeros(X.shape[0] * X.shape[1], dtype='bool')
    for i, p in enumerate(ps):
        logger.info(f"Processing path array: Poly {i}/{len(ps)} started.")
        flags = np.logical_or(p.contains_points(
            np.hstack((X.flatten()[:, np.newaxis], Y.flatten()[:, np.newaxis]))), flags)
    grid = np.zeros((h, w), dtype='bool')
    grid[:] = flags.reshape(h, w)
    return grid


class SquareCollision:

    def __init__(self, squares):
        """
        Instantiate a collider that checks for collision
        with arbitrary squares, normally for use as a goal
        area in the isochrone method.

        :param squares: Tuples of square coordinates in
            the format (x1, x2, y1, y2)
        """
        self.squares = squares

    @staticmethod
    def eval_square(sq, x, y):
        """
        Check which coordinates in the x-y arrays collide
        with a particular square.

        :param sq: The coordinates of the square, typically
            in the format (x1, x2, y1, y2)
        :param x: x-coordinates of 2-D target points.
        :param y: y-coordinates of 2-D target points.
        :return: Array of booleans representing which points
            have collided.
        """
        sq_0 = min(sq[0], sq[1])
        sq_1 = max(sq[0], sq[1])
        sq_2 = min(sq[2], sq[3])
        sq_3 = max(sq[2], sq[3])
        return np.logical_and(np.logical_and(sq_0 <= x, x <= sq_1), np.logical_and(sq_2 <= y, y <= sq_3))

    def evaluate(self, x, y):
        """
        For each point represented by the x-y arrays, check
        if it has collided with any square in this object.

        :param x: x-coordinates of 2-D target points.
        :param y: y-coordinates of 2-D target points.
        :return: Array of booleans representing which points
            have collided.
        """
        valid = np.zeros(x.shape, dtype=np.uint8)
        for s in self.squares:
            _v = self.eval_square(s, x, y)
            valid = np.logical_or(valid, _v)

        return valid

    def plot(self):
        for s in self.squares:
            plt.plot([s[0], s[0], s[1], s[1], s[0]], [
                     s[2], s[3], s[3], s[2], s[2]], c="green")


class GoalCollision:

    def __init__(self, goals):
        """
        Instantiate a collider that checks for collision
        through two-point goals, normally for use as a goal
        area in the isochrone method.

        :param goals: Tuples of goal coordinates in
            the format (x1, x2, y1, y2)
        """
        self.goals = goals
        self.goal_coords = []
        self.goal_coords_plot = []
        self.max_count = 1

    def adjust_ended_collider(self, world_map):
        for g in self.goals:
            g1, g2 = world_map.to_coords(
                np.array(g[0]), np.array(g[2]), to_int=False)
            g3, g4 = world_map.to_coords(
                np.array(g[1]), np.array(g[3]), to_int=False)

            line_coords = bresenhamline(
                np.array([[g1, g2]]), np.array([g3, g4]), -1).T
            can_collide = world_map.is_inside(line_coords[1], line_coords[0])
            line_coords = line_coords.T[can_collide].T
            line_coords = line_coords.astype(np.int32)
            line_coords = line_coords.T[np.logical_not(
                world_map.collision[line_coords[1], line_coords[0]])].T
            line_coords = line_coords.astype(np.float32)

            line_coords[0] -= world_map.collision.shape[1]/2
            line_coords[1] -= world_map.collision.shape[0]/2

            line_coords[0] /= world_map.collision.shape[1]/2
            line_coords[1] /= world_map.collision.shape[0]/2

            line_coords[0] *= 180
            line_coords[1] *= 90

            self.goal_coords_plot = line_coords

            for l in line_coords.T:
                self.goal_coords.append(Point(l[0], l[1]))

        self.max_count = len(self.goal_coords)

    @staticmethod
    def eval_goal(g, a, b):
        """
        Check if the segment defined by the points a, b
        cross through the goal g.

        :param g: The coordinates of the goal, typically
            in the format (x1, x2, y1, y2)
        :param a: origin point
        :param b: target point
        :return: Boolean representing if the segment a, b
            went through the goal
        """
        if (a[0] > 140 and b[0] < -140) or (b[0] > 140 and a[0] < -140):
            if a[0] > 140:
                i_a = intersect(
                    a + np.array([-360, 0]), b, np.array([g[0], g[2]]), np.array([g[1], g[3]]))
                i_b = intersect(
                    a, b + np.array([360, 0]), np.array([g[0], g[2]]), np.array([g[1], g[3]]))

                return i_a or i_b
            else:
                i_a = intersect(
                    a + np.array([360, 0]), b, np.array([g[0], g[2]]), np.array([g[1], g[3]]))
                i_b = intersect(
                    a, b + np.array([-360, 0]), np.array([g[0], g[2]]), np.array([g[1], g[3]]))

                return i_a or i_b

        return intersect(a, b, np.array([g[0], g[2]]), np.array([g[1], g[3]]))

    def evaluate(self, A, B, world_map):
        """
        Check if the segments defined by the point arrays
        A, B intersect with any of the goals included in
        the class instance.

        :param A: Array of initial points.
        :param B: Array of target points.
        :return: Array of booleans representing which points
            have collided.
        """
        valid = np.zeros(len(A), dtype=np.bool)
        for g in self.goals:
            _ = 0
            for a, b in zip(A, B):
                valid[_] = np.logical_or(self.eval_goal(g, a, b), valid[_])
                _ += 1

        return valid

    def clean(self, poly):
        self.goal_coords = [
            p for p in self.goal_coords if not poly.contains(p)]
        # self.goal_coords_plot = self.goal_coords_plot.T
        # self.goal_coords_plot = [p2 for p, p2 in zip(self.goal_coords, self.goal_coords_plot) if not poly.contains(p)]
        # self.goal_coords_plot = np.array(self.goal_coords_plot).T

    def plot(self):
        for s in self.goals:
            plt.plot(s[:2], s[2:], c="green")
            plt.scatter(s[:2], s[2:], c="green", s=5)
            # plt.scatter(self.goal_coords_plot[0], self.goal_coords_plot[1], color="orange")

    def get_count(self):
        return len(self.goal_coords)/self.max_count


class MapCollision:

    def __init__(self, bounds, north_limit, map=[18771, 9385]):
        """
        Instantiate a collider that checks for collision
        with world map collision, normally used as a
        boundary in the isochrone method.

        :param bounds: Bounds of the map coordinates
            (typically -180, 180, -90, 90)
        :param north_limit: The northern limit not to be
            crossed.
        """
        self._north_limit = north_limit
        self.north_limit = 1 - (self._north_limit + 90)/180

        self.x1, self.x2, self.y1, self.y2 = bounds

        if 1:
            import shapefile

            shapely_shapes = []
            from shapely.geometry import shape

            with shapefile.Reader("ne_10m_land/ne_10m_land.shp") as shp:
                shp = shp.shapeRecords()
                for s in shp:
                    s = s.shape.__geo_interface__
                    s = shape(s)
                    shapely_shapes.append(s)

            paths = []

            ctr = 0

            for s in shapely_shapes:
                if type(s) == MultiPolygon:
                    for _ in s:
                        paths.append(path.Path(_.exterior.coords))
                        # print(ctr, len(_.exterior.coords))
                        ctr += 1
                elif type(s) == Polygon:
                    paths.append(path.Path(s.exterior.coords))
                    # print(ctr, len(s.exterior.coords))
                    ctr += 1
        else:
            world = geopandas.read_file(
                geopandas.datasets.get_path('naturalearth_lowres'))

            for s in world.boundary:
                if type(s) == MultiLineString:
                    for _ in s:
                        paths.append(path.Path(_.coords))
                elif type(s) == LineString:
                    paths.append(path.Path(s.coords))

        cw, ch = map[0], map[1]
        self.cw, self.ch = cw, ch
        if not os.path.exists(f"worldarrays/{cw}x{ch}.npy"):
            collision = path_array(paths, cw, ch, (-180, 180, -90, 90))
            np.save(f"worldarrays/{cw}x{ch}.npy", collision)
        else:
            collision = np.load(f"worldarrays/{cw}x{ch}.npy")

        if not os.path.exists(f"worldarrays/{cw}x{ch}_south.npy"):
            import json
            with open('VG_south_limit.json') as json_file:
                data = json.load(json_file)
                data = data["south"]
                lats = [-90] + [d["lat"] for d in data] + [-90]
                lons = [-180] + [d["lon"] for d in data] + [180]
                coords = np.array([lons, lats]).T
                p = [path.Path(coords)]
            collision = np.logical_or(path_array(
                p, cw, ch, (-180, 180, -90, 90)), collision)
            np.save(f"worldarrays/{cw}x{ch}_south.npy", collision)
        else:
            collision = np.load(f"worldarrays/{cw}x{ch}_south.npy")

        self.north_limit = int(self.north_limit * ch)

        self.collision = collision
        self.collision[-self.north_limit:] = True

        self.ch, self.cw = self.collision.shape

    def is_inside(self, x, y):

        valid = np.zeros(x.shape, dtype=np.bool)

        valid = np.logical_or(valid, x < 0)
        valid = np.logical_or(valid, x > self.collision.shape[0])
        valid = np.logical_or(valid, y < 0)
        valid = np.logical_or(valid, y > self.collision.shape[1])

        return np.logical_not(valid)

    def recollide(self, poly):
        """
        Use an additional collection of polygons to constrain
        allowable areas to the intersection of the allowable
        area on the world map as well as the allowable area
        defined by the polygons.

        :param poly: Multipolygon or list of polygons.
        """
        collision = path_array([path.Path(p.exterior.coords)
                                for p in poly], self.cw, self.ch, (-180, 180, -90, 90))
        darkener = np.logical_not(collision)
        self.collision = np.logical_or(
            darkener, self.collision).reshape(self.ch, self.cw)

    def recollide_squares(self, points, radius):
        radius = self.to_coords(np.array(radius-180), np.array(radius-90))[0]

        x, y = self.to_coords(*points.T)
        points = np.vstack((x, y)).T

        self.collision_ = np.ones(self.collision.shape, dtype='bool')
        for p in points:

            x_lower = int(p[0] - radius)
            x_upper = int(p[0] + radius)
            y_lower = int(p[1] - radius)
            y_upper = int(p[1] + radius)

            self.collision_[y_lower:y_upper, x_lower:x_upper] = False

        self.collision = np.logical_or(self.collision, self.collision_)

    def to_coords(self, x, y, to_int=True):
        """
        Modifies the lat-long x-y coordinates to be the
        array coordinates.

        :param x: x-coordinate array
        :param y: y-coordinate array
        :return: The converted coordinates to index into
            the world map collision array.
        """
        _x = (x + 180) / 360
        _y = (y + 90) / 180
        _x *= self.cw
        _y *= self.ch
        if to_int:
            _x = _x.astype(np.int32)
            _y = _y.astype(np.int32)

            _x = np.clip(_x, 0, self.cw - 1)
            _y = np.clip(_y, 0, self.ch - 1)

        return _x, _y

    def eval_map(self, x, y):
        """
        For a collection of x-y coordinates, check if they
        are colliding in the collision array.

        :param x: x-coordinate array
        :param y: y-coordinate array
        :return: boolean array defining if a point has collided.
        """

        _x = (x + 180)/360
        _y = (y + 90)/180
        _x *= self.cw
        _y *= self.ch
        _x = _x.astype(np.int32)
        _y = _y.astype(np.int32)

        _x = np.clip(_x, 0, self.cw-1)
        _y = np.clip(_y, 0, self.ch-1)

        bls = self.collision[_y, _x]
        return bls

    def evaluate(self, x, y):
        """
        For a collection of x-y coordinates, check if they
        are colliding in the collision array with boundary.

        :param x: x-coordinate array
        :param y: y-coordinate array
        :return: boolean array defining if a point has collided.
        """
        valid = self.eval_map(x, y)

        valid = np.logical_or(valid, x < self.x1)
        valid = np.logical_or(valid, x > self.x2)
        valid = np.logical_or(valid, y < self.y1)
        valid = np.logical_or(valid, y > self.y2)

        return valid

    def plot(self):
        plt.imshow(np.flip(self.collision, axis=0),
                   extent=(-180, 180, -90, 90), cmap="Greys")
        plt.hlines(self._north_limit, -180, 180, color="red")


class Isochrone:

    def __init__(self, delta_angle, envelope_func, vector_field, north_limit, collision_target2, hour_scale=6., check_intersect=False, iso_link=None, map=[18771, 9385]):
        """
        Single-use object for computing the best maritime path
        from point A to point B under certain wind conditions
        using the isochrone method.

        :param delta_angle: Angle step value between each relative
            path from a single point to consider.
        :param envelope_func: Function used to compute velocity
            envelopes relative to a certain wind speed and direction.
        :param vector_field: Function used to compute vector
            components based on positions in space and time.
        :param hour_scale: Time step value (in hours) for the delta
            between each isochrone.
        :param check_intersect: Defines whether to check linear
            intersections for paths, leave False for faster but less
            reliable path calculations.
        :param north_limit: The northern limit not to be
            crossed.
        """
        bounds = -180, 180, -90, 90

        self.da = delta_angle
        self.super_polygon = None

        self.collision_polys = []
        self.collision_polylines = []

        self.found = False
        self.win_path = None

        self.hour_scale = hour_scale
        self.check_intersect = check_intersect

        self.collision_target2 = collision_target2
        self.north_limit = north_limit
        self.collision_map = MapCollision(
            (-180, 180, -90, 90), north_limit, map)

        self.collision_target2.adjust_ended_collider(self.collision_map)

        self.envelope_function = envelope_func
        self.wind_vector_field = vector_field
        self.lons, self.lats = [], []

        self.x1, self.x2, self.y1, self.y2 = bounds

        self.it = 0
        self.history = []
        self.history_id = []

        self.ox, self.oy = None, None
        self.iso_link = iso_link

        self.index_link = {}
        self.parent = None

    def propagate_goals(self, goal_array):
        prev_iso = None
        for g in reversed(goal_array):
            prev_iso = Isochrone(self.da, self.envelope_function, self.wind_vector_field, self.north_limit,
                                 g, hour_scale=self.hour_scale, check_intersect=self.check_intersect, iso_link=prev_iso)
        self.iso_link = prev_iso

    def pre_compute(self, x, y, bounds):
        """
        Call this once before computing isochrones, sets the
        current considered points to x and y.

        :param x: start x-coordinate
        :param y: start y-coordinate
        """
        self.ox, self.oy = x, y
        self.lons, self.lats = [x], [y]
        self.collision_map.x1, self.collision_map.x2, self.collision_map.y1, self.collision_map.y2 = bounds

        if self.iso_link is not None:
            self.iso_link.parent = self
            self.iso_link.pre_compute(0, 0, bounds)
            self.iso_link.lats = []
            self.iso_link.lons = []

    def clean(self, op, op2):
        """
        Removes out-of-bounds points from the current considered
        points.
        """
        t = time.time_ns()
        xs, ys = self.lons, self.lats
        self.history_id[-1] = np.array(self.history_id[-1])

        valid = self.collision_map.evaluate(xs, ys)
        valid = np.logical_not(valid)
        self.history_id[-1] = self.history_id[-1][valid]
        xs = xs[valid]
        ys = ys[valid]

        logger.debug(f"clean:{(time.time_ns() - t)/1e9}")

        self.lons, self.lats = xs, ys

        return op[valid], op2[valid]

    def get_poly_coords(self):
        """
        Get the coordinate lists for the polygon(s) defining
        the outer isochrone.

        :return: coordinate lists for the polygon(s) defining
        the outer isochrone.
        """
        if type(self.super_polygon) == Polygon:
            return self.super_polygon.exterior.coords.xy
        elif type(self.super_polygon) == MultiPolygon:
            xxs, yys = [], []
            for p in self.super_polygon:
                xs, ys = p.exterior.coords.xy
                xxs.append(xs)
                yys.append(ys)
            xs, ys = np.concatenate(xxs), np.concatenate(yys)
            return xs, ys

    def get_poly_plot(self):
        """
        Get the coordinate lists for the polygon(s) defining
        the outer isochrone (Adjusted for plotting).

        :return: coordinate lists for the polygon(s) defining
        the outer isochrone (Adjusted for plotting).
        """
        if type(self.super_polygon) == Polygon:
            return [self.super_polygon.exterior.coords.xy[0]], [self.super_polygon.exterior.coords.xy[1]]
        elif type(self.super_polygon) == MultiPolygon:
            xxs, yys = [], []
            for p in self.super_polygon:
                xs, ys = p.exterior.coords.xy
                xxs.append(xs)
                yys.append(ys)
            return xxs, yys

    def refocus(self, op, op2):
        """
        Sets the current considered points to ONLY points
        included in the current outer envelope polygon.

        :param op: Points array to be filtered.
        """
        t = time.time_ns()
        xs, ys = self.get_poly_coords()

        pts = np.vstack((xs, ys)).T
        lst = []
        lst2 = []
        n_hist = []

        for p, p2, h in zip(op, op2, self.history_id[-1]):
            if np.equal(pts, p).all(1).any():
                lst.append(p)
                lst2.append(p2)
                n_hist.append(h)

        self.history_id[-1] = n_hist
        self.lons, self.lats = np.array(lst).T
        logger.debug(f"refocus:{(time.time_ns() - t)/1e9}")

        return np.array(lst), np.array(lst2)

    def backtrack(self, t, i):
        """
        Returns the backwards path from the i^th point for
        the t^th isochrone.

        :param t: The isochrone to be considered backtracking
            from.
        :param i: The index of the point to be backtracked from.
        :return: Backtracked path.
        """
        path = []
        while t > -1:
            try:
                pt = self.history[t][i]
                i = self.history_id[t][i]
                t -= 1
                path.append(pt)

                if self.parent is not None and (pt[0], pt[1]) in self.parent.index_link:
                    i, l = self.parent.index_link[(pt[0], pt[1])]
                    path = path + self.parent.backtrack(l - 1, i)
                    break
            except Exception as e:
                print(e)
                break

        return path

    @staticmethod
    def gen_extra(pts):
        """
        Generate extra points depending on the number of
        points in pts, until we have three valid points
        for a small polygon.

        :param pts: Points array.
        :return: Augmented points array.
        """
        ret = [p for p in pts]
        mean = np.mean(pts, axis=0)
        var = np.var(pts, axis=0)/1000

        while len(ret) < 3:
            ret.append(np.random.normal(mean, var, 2))

        return np.array(ret)

    def compute(self, sail_time=0., prev_collide=None):
        """
        Compute the next isochrone at time z.

        :param sail_time: Time at which the next isochrone is
            computed, used to calculate wind velocities.
        :return: x-y coordinate arrays for current valid
            points.
        """
        if len(self.lons) == 0:
            if self.iso_link is not None:
                self.iso_link.compute(
                    sail_time=sail_time, prev_collide=self.collision_target2)
            return np.array([]), np.array([])

        if self.collision_target2.get_count() < 0.04:
            if self.iso_link is not None:
                self.iso_link.compute(
                    sail_time=sail_time, prev_collide=self.collision_target2)
            return np.array([]), np.array([])

        logger.debug(f"Amount of points:{len(self.lons)}")
        self.it += 1
        if len(self.lons) == 0:
            return [], []

        t = time.time_ns()
        dx, dy = self.wind_vector_field._compute(
            self.lons, self.lats, z=sail_time)
        logger.debug(f"winds:{(time.time_ns() - t)/1e9}")

        t = time.time_ns()
        pts = self.envelope_function.compute(
            dx, dy, self.lons, self.lats, self.da, scale=self.hour_scale)
        logger.debug(f"envelopes:{(time.time_ns() - t)/1e9}")

        logger.debug(f"pts shape:{pts.shape}")

        self.history_id.append([])

        polys = []

        t = time.time_ns()

        sub_check = []

        for p, x, y in zip(pts, self.lons, self.lats):
            s_lim = 160
            if x > s_lim:
                p_ = p.T
                p1 = p_[0] > 0
                p2 = p_[0] < 0
                if self.check_intersect:
                    sub_check.append(p1)

                p_mod = np.copy(p_)
                p_mod[0][p2] += 360
                polys.append(Polygon(p_mod.T))

                p1 = p[p1]
                p2 = p[p2]

                if len(p2) > 0:
                    if len(p2) < 3:
                        polys.append(Polygon(self.gen_extra(p2)))
                    else:
                        polys.append(Polygon(p2))

            elif x < -s_lim:
                p_ = p.T
                p1 = p_[0] > 0
                p2 = p_[0] < 0
                if self.check_intersect:
                    sub_check.append(p2)

                p_mod = np.copy(p_)
                p_mod[0][p1] -= 360
                polys.append(Polygon(p_mod.T))

                p1 = p[p1]
                p2 = p[p2]
                if len(p1) > 0:
                    if len(p1) < 3:
                        polys.append(Polygon(self.gen_extra(p1)))
                    else:
                        polys.append(Polygon(p1))
            else:
                if self.check_intersect:
                    sub_check.append(np.array(([True for _ in p])))
                polys.append(Polygon(p))

        logger.debug(f"polys:{(time.time_ns() - t)/1e9}")

        n = 0
        all_pts = []
        og_pts = []

        t = time.time_ns()
        ctr = -1
        for p, x, y in zip(pts, self.lons, self.lats):
            ctr += 1
            try:
                if not self.check_intersect:
                    self.history_id[-1] += [n for _ in p]
                    all_pts.append(p)
                else:
                    f_p = []
                    nx, ny = self.collision_map.to_coords(x, y)
                    xy = np.array([nx, ny])
                    for i, _p in enumerate(p):
                        if not sub_check[ctr][i]:
                            f_p.append(_p)
                            continue

                        npx, npy = self.collision_map.to_coords(_p[0], _p[1])
                        pxy = np.array([[npx, npy]])

                        isect = bresenhamline(pxy, xy, -1).T
                        if np.any(self.collision_map.collision[isect[1], isect[0]]):
                            continue
                        f_p.append(_p)

                    if len(f_p) == 0:
                        n += 1
                        continue

                    self.history_id[-1] += [n for _ in f_p]
                    all_pts.append(f_p)
                    og_pts.append([[x, y] for _ in f_p])

            except ValueError:
                pass
            n += 1
        logger.debug(f"histories:{(time.time_ns() - t)/1e9}")

        all_pts = np.vstack(all_pts)
        og_pts = np.vstack(og_pts)

        t = time.time_ns()

        if self.super_polygon is not None and type(self.super_polygon) == MultiPolygon:
            self.super_polygon = cascaded_union(
                [p for p in self.super_polygon])

        if self.super_polygon is None:
            self.super_polygon = cascaded_union(polys)
        else:
            n_threads = 12
            polys = chunks(polys, n_threads)
            pool = ThreadPool(n_threads)
            polys = pool.map(self.poly_union, polys)
            pool.close()
            pool.join()

            self.super_polygon = cascaded_union([
                geom if geom.is_valid else geom.buffer(0) for geom in polys + [self.super_polygon]
            ])

        logger.debug(f"polystack:{(time.time_ns() - t)/1e9}")

        self.lons, self.lats = all_pts.T

        if len(self.lons) == 0:
            return

        all_pts, og_pts = self.refocus(all_pts, og_pts)
        all_pts, og_pts = self.clean(all_pts, og_pts)

        reached = self.collision_target2.evaluate(
            og_pts, all_pts, self.collision_map)
        if np.any(reached):
            self.win_path = np.arange(
                0, len(self.history_id[-1]), 1)[reached][-1]
            self.found = True
            logger.debug("Isochrones finishing, found winning path.")

        ap_lons = None
        ap_lats = None
        ap_hist = None
        ap_hist_id = None

        if self.iso_link is not None:
            if np.any(reached):
                not_reached = np.logical_not(reached)

                if self.iso_link is not None:
                    ap_lons = self.lons[reached]
                    ap_lats = self.lats[reached]
                    ap_hist = np.vstack(
                        (self.lons[reached], self.lats[reached])).T
                    ap_hist_id = np.array(
                        [-1 for _ in range(len(self.lons[reached]))])

                for lon, lat, h in zip(self.lons[reached], self.lats[reached], self.history_id[-1][reached]):
                    self.index_link[(lon, lat)] = h, len(self.history)

                self.lons = self.lons[not_reached]
                self.lats = self.lats[not_reached]

                og_pts = og_pts[not_reached]
                all_pts = all_pts[not_reached]

                self.history_id[-1] = self.history_id[-1][not_reached]

        if not (self.found and self.iso_link is None):
            if prev_collide is not None:
                reached = prev_collide.evaluate(
                    og_pts, all_pts, self.collision_map)

                if np.any(reached):
                    not_reached = np.logical_not(reached)

                    self.lons = self.lons[not_reached]
                    self.lats = self.lats[not_reached]

                    self.history_id[-1] = self.history_id[-1][not_reached]

        self.history.append(np.vstack((self.lons, self.lats)).T)

        if self.iso_link is not None:
            self.iso_link.compute(sail_time=sail_time,
                                  prev_collide=self.collision_target2)

        if ap_lons is not None:
            self.iso_link.lons = np.concatenate((self.iso_link.lons, ap_lons))
            self.iso_link.lats = np.concatenate((self.iso_link.lats, ap_lats))
            if len(self.iso_link.history) == 0:
                self.iso_link.history.append(ap_hist)
            else:
                self.iso_link.history[-1] = np.vstack(
                    (self.iso_link.history[-1], ap_hist))

            if len(self.iso_link.history_id) == 0:
                self.iso_link.history_id.append(ap_hist_id)
            else:
                self.iso_link.history_id[-1] = np.concatenate(
                    (self.iso_link.history_id[-1], ap_hist_id))

        self.collision_target2.clean(self.super_polygon)

        return self.lons, self.lats

    @staticmethod
    def check_traversed_goals(goal_coords, path_coords, north_limit):
        """
        :param goal_coords: (((x1, x2, y1, y2), (...)), ...)
        :param path_coords: ((x, y), (x, y), ...)
        :return:
        """

        goals = [GoalCollision(g) for g in goal_coords]
        path_coords = np.array(path_coords)
        p1 = path_coords[:-1]
        p2 = path_coords[1:]

        cl = MapCollision((-180, 180, -90, 90), north_limit)

        ret = [np.any(g.evaluate(p1, p2, cl)) for g in goals]

        return ret

    @staticmethod
    def check_collision(start_coord, possible_coords, north_limit):
        """

        :param start_coord: [x, y]
        :param possible_coords: [[x1, y1], [x2, y2], ...]
        :param north_limit:
        :return:
        """
        start_coord = np.array(start_coord)
        cl = MapCollision((-180, 180, -90, 90), north_limit)
        x, y = start_coord
        ret = []
        nx, ny = cl.to_coords(x, y)
        xy = np.array([nx, ny])
        for i, _p in enumerate(possible_coords):
            _p = np.array(_p)
            npx, npy = cl.to_coords(_p[0], _p[1])
            pxy = np.array([[npx, npy]])

            isect = bresenhamline(pxy, xy, -1).T
            ret += [np.any(cl.collision[isect[1], isect[0]])]

        return ret

    @staticmethod
    def poly_union(polys):
        """
        Make a union of polygons, checking for invalid ones.

        :param polys: Polygons to make a union of.
        :return: Unioned poly/multipolygon object.
        """
        return cascaded_union([
            geom if geom.is_valid else geom.buffer(0) for geom in polys
        ])

    def random_backtrack(self):
        """
        Get a random path back from the last isochrone.

        :return:
        """
        rr = np.random.randint(len(self.history[-1]))
        return self.backtrack(len(self.history)-1, rr)

    def bct(self, i):
        """
        Backtrack in order to find the path from the
        last isochrone's point with index i.

        :param i: Index of point to backtrack from.
        :return: Backtracked path.
        """
        return self.backtrack(len(self.history) - 1, i)

    def winning_poly(self, radius):
        """
        Get the polygon that surrounds the winning path
        with radius radius.

        :param radius: Radius of polygon around the winning
            path.
        :return: Polygon around the winning path.
        """
        j = self.win_path
        pts = np.array(self.bct(j))
        return self.surround_poly(pts, radius)

    def surround_poly(self, pts, radius):
        """
        Get the polygon surrounding the points pts by
        radius radius.

        :param pts: Points to surround.
        :param radius: Radius of polygon around the winning
            path.
        :return: Polygon around the points pts.
        """
        polys = []

        for p in pts:
            theta = 0.1
            _ = np.array([radius, 0])

            mat = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            surround = []

            ctr = 0
            while ctr < np.pi * 2:
                surround.append(np.copy(_))
                _ = mat.dot(_)
                ctr += theta

            surround = np.array(surround)

            surround += p

            polys.append(Polygon(surround))

        ret = cascaded_union(polys)

        if type(ret) != MultiPolygon:
            ret = [ret]

        return ret

    def plot(self, isochrones_current, bounds, sail_time):
        plt.clf()

        for _X, _Y in self.isochrones_history:
            plt.plot(_X, _Y, linewidth=1, c="gray", alpha=0.4)

        import matplotlib
        norm = matplotlib.colors.Normalize(-1, 1)
        colors = [
            [norm(-1), "#000066"],
            [norm(-0.33), "#0000ff"],
            [norm(0.33), "#ff0000"],
            [norm(1), "#660000"],
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

        x1, x2, y1, y2 = bounds
        X, Y = np.meshgrid(np.linspace(x1, x2, 45), np.linspace(y1, y2, 45))
        self.wind_vector_field.pre_show(X, Y, z=sail_time, cmap=cmap)

        for xy in isochrones_current:
            plt.plot(xy[0], xy[1], linewidth=1, c="red")

        self.collision_map.plot()
        self.collision_target2.plot()

    def iso_plot(self, bounds, sail_time, partial=False):

        isochrones_current = []

        target = self

        if len(self.lons) != 0 and target.get_poly_plot() is not None:
            xpp, ypp = target.get_poly_plot()

            for xp, yp in zip(xpp, ypp):
                XY = np.vstack((xp, yp))

                if len(XY) == 0:
                    continue

                self.isochrones_history.append(XY)
                isochrones_current.append(XY)

        if not partial:
            target.plot(isochrones_current, bounds, sail_time)
        else:
            target.partial_plot(isochrones_current, color="blue")

        if self.iso_link is not None:
            self.iso_link.iso_plot(bounds, sail_time, partial=True)

        plt.xlim(bounds[0] - 1, bounds[1] + 1)
        plt.ylim(bounds[2] - 1, bounds[3] + 1)

    def partial_plot(self, isochrones_current, color="red"):

        for _X, _Y in self.isochrones_history:
            plt.plot(_X, _Y, linewidth=1, c="gray", alpha=0.4)

        for xy in isochrones_current:
            plt.plot(xy[0], xy[1], linewidth=1, c=color)

        self.collision_target2.plot()

    def recollide_map(self, constrain_path, constrain_radius):
        self.collision_map.recollide_squares(constrain_path, constrain_radius)
        if self.iso_link is not None:
            self.iso_link.recollide_map(constrain_path, constrain_radius)

    def propagate(self):
        self.isochrones_history = []
        if self.iso_link is not None:
            self.iso_link.propagate()

    def last_iso(self):
        if self.iso_link is not None:
            return self.iso_link.last_iso()
        else:
            return self

    def full_compute(self, start_lon, start_lat, request_id, constrain_path=None, constrain_radius=10., ratio_delta_h=1, plot_result=False, start_time=0, bounds=(-180, 180, -90, 90)):
        """
        Do a full computation of the Isochrone object and return
        a winning path.

        :param start_lon: x-coordinate of starting point
        :param start_lat: y-coordinate of starting point
        :param constrain_path: Polygon to constrain the search in
        :param constrain_radius: Radius of the polygon to constrain the search in
        :param ratio_delta_h: divide the delta hour by this parameter (introduced due to a mismatch between ETA predictions & actual passing time during the Vendée Globe)
        :param plot_result: Whether or not to display a running
            plot
        :param start_time: How many hours after the last data update
            the race is starting
        :param bounds: The region to consider when running the iso
            algorithm.
        :return: The winning path, and the time delta for each point
            in the winning path
        """
        if ratio_delta_h != 1:
            # Right now, this is expected only for Vendée Globe
            logger.warning(
                f'Request ID {request_id}: Ratio delta hour is different than 1, hence the delta hour will be divided by its value: {ratio_delta_h}. Are you sure this is expected behavior?')

        delta_hours = self.hour_scale

        o_time = time.time_ns()
        if constrain_path is not None:
            self.recollide_map(constrain_path, constrain_radius)

        self.pre_compute(start_lon, start_lat, bounds)

        times = []

        self.propagate()

        time_points = []

        isochrone_step = 0
        while not self.last_iso().found:
            logger.debug(f'Last iso found? {self.last_iso().found}')
            sail_time = isochrone_step * delta_hours/ratio_delta_h + start_time
            if len(self.lons) == 0:
                logger.warning(
                    f'Request ID {request_id}: Could not find winning path. This may be because the initial point started on land.')
                logger.warning(
                    f'Request ID {request_id}: additional info: Isochrone {isochrone_step}, Current time:{sail_time}')
                pass

            time_points.append(sail_time - start_time + delta_hours)
            c_time = time.time_ns()
            logger.debug(f"Current time:{sail_time}")
            self.compute(sail_time=sail_time)

            times.append(time.time_ns() - c_time)

            logger.debug(
                f"Isochrone {isochrone_step} took {times[-1] / 1e9} seconds.")

            if plot_result:
                self.iso_plot(bounds, sail_time)

                plt.draw()
                plt.pause(1e-6)

            isochrone_step += 1

        if plot_result:
            self.iso_plot(bounds, sail_time)

            j = self.last_iso().win_path
            pts = np.array(self.last_iso().bct(j)).T
            skip = pts[0][:-1] - pts[0][1:]
            skip = np.logical_or(skip < -180, skip > 180)
            skip = np.concatenate((np.array([False]), skip))
            pts[0][skip] = np.nan
            pts[1][skip] = np.nan
            plt.plot(pts[0], pts[1], c="orange", linewidth=2, alpha=1)

            plt.show()
            plt.draw()
            plt.pause(1e-6)

        logger.debug(f"Total time: {(time.time_ns() - o_time) / 1e9}")

        ret_path = np.flip(
            np.array(self.last_iso().bct(self.last_iso().win_path)), axis=0)

        time_points = (np.arange(len(ret_path)) + 1) * self.hour_scale

        return ret_path, time_points
