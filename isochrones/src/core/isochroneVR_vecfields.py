import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.common.log import logger


class WindField:

    def __init__(self, request_id, process_winds=True):
        """
        A vector field that utilises wind data from the
        NOAA, and can be used to get a time and space-wise
        smoothly interpolated wind direction vector for
        any arbitrary point in space and time.


        :param bounds: The bounds of the map area, typically
            (-180, 180, -90, 90)
        :param process_winds: Do we reprocess and re-load the
            files for the winds? Should be yes when we have
            downloaded new wind files.
        """
        bounds = -180, 180, -90, 90

        self.data_collection = {}
        filelist = [f.split(".")[0]
                    for f in os.listdir(f"{request_id}/winds") if f.endswith(".csv")]
        max_file = max([int(i) for i in filelist])

        filelist.append(str(int(1e9)))

        if process_winds:
            from shutil import copyfile
            copyfile(f"{request_id}/winds/{max_file}.csv",
                     f"{request_id}/winds/{int(1e9)}.csv")

            for i in filelist:
                data = pd.read_csv(f"{request_id}/winds/{i}.csv")
                data = data.dropna()

                Y, X, U, V = data.to_numpy(dtype=np.float32).T
                X = X.astype(np.int32)
                Y = Y.astype(np.int32)

                c_U = np.zeros((181, 360), dtype=np.float32)
                c_V = np.copy(c_U)
                c_D = np.copy(c_U)
                c_Theta = np.copy(c_U)

                uv = np.vstack((U, V)).T
                inv = np.degrees(np.arctan2(*uv.T[::-1])) % 360.0

                for x, y, u, v, theta in zip(X, Y, U, V, inv):
                    c_U[y + 90, x + 180] = u
                    c_V[y + 90, x + 180] = v
                    c_D[y + 90, x + 180] = np.sqrt(u**2 + v**2)
                    c_Theta[y + 90, x + 180] = theta

                self.data_collection[int(i)] = [c_U, c_V, c_D, c_Theta]

                np.save(f"{request_id}/winds_processed/{i}_U.npy", c_U)
                np.save(f"{request_id}/winds_processed/{i}_V.npy", c_V)
                np.save(f"{request_id}/winds_processed/{i}_D.npy", c_D)
                np.save(f"{request_id}/winds_processed/{i}_Theta.npy", c_Theta)

        else:
            logger.info("Skipped wind preprocessing, loading from disk.")
            for i in filelist:
                c_U = np.load(f"{request_id}/winds_processed/{i}_U.npy")
                c_V = np.load(f"{request_id}/winds_processed/{i}_V.npy")
                c_D = np.load(f"{request_id}/winds_processed/{i}_D.npy")
                c_Theta = np.load(
                    f"{request_id}/winds_processed/{i}_Theta.npy")

                self.data_collection[int(i)] = [c_U, c_V, c_D, c_Theta]

        self.boat_plot = []
        self.boat_plotp = []

        self.x1, self.x2, self.y1, self.y2 = bounds

    def get_map(self, t):
        """
        Get the time-interpolated wind map from the two wind
        maps surrounding the point t in time.

        :param t: The time for which to get our wind map.
        :return: The u-v components of the time-interpolated
            vectors, as well as a distance-angle representation.
        """
        collection_times = list(self.data_collection.keys())
        time_lower = max([c for c in collection_times if c <= t])
        time_upper = min([c for c in collection_times if c > t])
        logger.debug(f"Times: {time_lower}, {time_upper}")
        time_lower_map = self.data_collection[time_lower]
        time_upper_map = self.data_collection[time_upper]

        U_lower, V_lower = time_lower_map[:2]
        U_upper, V_upper = time_upper_map[:2]

        t_diff = t - time_lower
        t_diff_plus = time_upper - time_lower

        U_diff = U_upper - U_lower
        V_diff = V_upper - V_lower

        Un = U_lower + t_diff * (U_diff / t_diff_plus)
        Vn = V_lower + t_diff * (V_diff / t_diff_plus)

        Dn = np.zeros(Un.shape)
        Tn = np.zeros(Un.shape)

        for i in range(Dn.shape[0]):
            for j in range(Dn.shape[1]):
                u, v = Un[i, j], Vn[i, j]
                theta = np.degrees(np.arctan2(u, v)) % 360
                norm = np.sqrt(u**2 + v**2)
                Dn[i, j] = norm
                Tn[i, j] = theta

        return Un, Vn, Dn, Tn

    def _compute(self, x, y, z=0, get_sizes=False):
        """
        For given x-y arrays and time z, compute the corresponding
        u-v components of wind vectors at their locations.

        :param x: x-coordinate array
        :param y: y-coordinate array
        :param z: time scalar
        :param get_sizes: if set to true, also return the sizes of
            the vectors
        :return: u-v components of vectors at (x, y, z)
        """

        x = np.array(x).flatten() + 180
        y = np.array(y).flatten() + 90

        x1_dist = x % 1.
        x2_dist = 1. - x1_dist
        y1_dist = y % 1.
        y2_dist = 1. - y1_dist

        x1, x2, y1, y2 = np.floor(x), np.floor(
            x) + 1, np.floor(y), np.floor(y) + 1
        x1, x2, y1, y2 = x1.astype(np.int32), x2.astype(
            np.int32), y1.astype(np.int32), y2.astype(np.int32)

        x1 %= 360
        x2 %= 360
        y1 %= 180
        y2 %= 180

        U, V, D, Theta = self.get_map(z)

        u1 = U[y1, x1]
        u2 = U[y1, x2]
        u3 = U[y2, x1]
        u4 = U[y2, x2]

        v1 = V[y1, x1]
        v2 = V[y1, x2]
        v3 = V[y2, x1]
        v4 = V[y2, x2]

        d1 = D[y1, x1]
        d2 = D[y1, x2]
        d3 = D[y2, x1]
        d4 = D[y2, x2]

        Theta1 = Theta[y1, x1]
        Theta2 = Theta[y1, x2]
        Theta3 = Theta[y2, x1]
        Theta4 = Theta[y2, x2]

        uc1 = (1 - x1_dist) * u1 + (1 - x2_dist) * u2
        uc2 = (1 - x1_dist) * u3 + (1 - x2_dist) * u4
        uc = uc1 * (1 - y1_dist) + uc2 * (1 - y2_dist)

        vc1 = (1 - x1_dist) * v1 + (1 - x2_dist) * v2
        vc2 = (1 - x1_dist) * v3 + (1 - x2_dist) * v4
        vc = vc1 * (1 - y1_dist) + vc2 * (1 - y2_dist)

        speed1 = (1 - x1_dist) * d1 + (1 - x2_dist) * d2
        speed2 = (1 - x1_dist) * d3 + (1 - x2_dist) * d4
        speed = speed1 * (1 - y1_dist) + speed2 * (1 - y2_dist)

        UV = np.vstack((uc, vc))

        uv_norm = np.sqrt(uc**2 + vc**2)
        UV_ = UV/uv_norm

        Theta13 = Theta1 - Theta3
        Theta24 = Theta2 - Theta4
        Theta12 = Theta1 - Theta2
        Theta34 = Theta3 - Theta4

        sumU = u1 + u2 + u3 + u4
        sumV = v1 + v2 + v3 + v4

        avg_w = np.sqrt((sumU/4)**2 + (sumV/4)**2)
        avg_s = (d1 + d2 + d3 + d4)/4

        f_center = avg_w/avg_s
        f_center[avg_s == 0] = 1

        def factd(deg): return np.abs(np.sin(deg/180 * np.pi))

        full_factor = 1/factd(90)

        fs = [
            [[full_factor for _ in Theta12], factd(
                Theta12), [full_factor for _ in Theta12]],
            [factd(Theta13), f_center, factd(Theta24)],
            [[full_factor for _ in Theta12], factd(
                Theta34), [full_factor for _ in Theta12]]
        ]
        fs = np.array(fs)

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        fx1 = np.zeros(len(mid_x), dtype=np.int32)
        fy1 = np.zeros(len(mid_y), dtype=np.int32)
        fx1[x >= mid_x] = 1
        fy1[y >= mid_y] = 1
        fx2 = 1 + fx1
        fy2 = 1 + fy1

        x1_dist_p = np.copy(x1_dist)
        x2_dist_p = np.copy(x2_dist)
        y1_dist_p = np.copy(y1_dist)
        y2_dist_p = np.copy(y2_dist)

        x1_dist_p[fx1 == 1] -= 0.5
        x2_dist_p[np.logical_not(fx1 == 1)] -= 0.5

        y1_dist_p[fy1 == 1] -= 0.5
        y2_dist_p[np.logical_not(fy1 == 1)] -= 0.5

        ponder_factors_1 = fs[fx1, fy1, np.arange(len(fx1))]
        ponder_factors_2 = fs[fx1, fy2, np.arange(len(fx1))]
        ponder_factors_3 = fs[fx2, fy1, np.arange(len(fx1))]
        ponder_factors_4 = fs[fx2, fy2, np.arange(len(fx1))]

        pp1 = ponder_factors_1 * (0.5 - y1_dist_p) + \
            ponder_factors_2 * (0.5 - y2_dist_p)
        pp2 = ponder_factors_3 * (0.5 - y1_dist_p) + \
            ponder_factors_4 * (0.5 - y2_dist_p)

        pp = pp1 * x1_dist_p + pp2 * x2_dist_p

        final = (uv_norm/speed)**(1 - pp**0.7)
        final[np.isnan(final)] = 1.

        f_speed = final * speed

        UV_ = UV_ * f_speed

        vc, uc = UV_

        if not get_sizes:
            return uc, vc
        else:
            return uc, vc, np.sqrt(uc**2 + vc**2)

    def pre_show(self, X, Y, z=0, cmap="coolwarm"):
        """
        Plot u-v components of wind vectors at (X, Y, z)

        :param X: x-coordinate array
        :param Y: y-coordinate array
        :param z: time scalar
        :param cmap: colormap used to paint the vectors
            by magnitude
        """
        Xp, Yp, s = self._compute(X, Y, z=z, get_sizes=True)
        plt.scatter([self.x1, self.x1, self.x2, self.x2], [
                    self.y1, self.y2, self.y1, self.y2], c="k", marker="D", s=100)
        plt.plot([self.x1, self.x1, None, self.x2, self.x2], [
                 self.y1, self.y2, None, self.y1, self.y2], c="k")
        plt.quiver(X.flatten(), Y.flatten(), Xp.flatten(),
                   Yp.flatten(), s, cmap=cmap)

        plt.tight_layout()
