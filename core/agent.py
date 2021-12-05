import math
import numpy as np
from enum import Enum
from loguru import logger
from scipy import interpolate

from .mess import Conversion, MESS, NavMesh
from .pf import PathPlanner


class Status(Enum):
    SUCCESS = 1
    REACHED = 2


class Agent:
    def __init__(
        self,
        start,
        goal,
        local_size,
        world,
        transform,
        agent_extent_bias="inverse_diagonal",
        recalculate_threshold=0.2,
    ) -> None:
        self.start = (*start, world[start[1], start[0]])
        self.goal = (*goal, world[goal[1], goal[0]])
        self.goal_in_sight = False

        # * Map related
        self.local_size = local_size
        self.world = world
        self.world_extent = Conversion.extent_to_poly(
            [0, world.shape[1], 0, world.shape[0]]
        )
        self.transform = transform
        self.agent_extent_bias = agent_extent_bias
        self.recalculate_threshold = recalculate_threshold

        # * Position related
        self.position = self.start
        self.path = np.array([self.position])
        self.previous_position = self.position

        # * MESS
        self.mess = None

        # * NavMesh
        self.navmesh = None

    def forward(self):
        # * This returns current observable polygon
        logger.info("Calculating current extent")
        poly = self.__calc_poly()

        # * Observe
        dem = self.__observe(poly)

        if self.mess is None:
            self.mess = MESS()
            self.mess.process(dem, poly)
        else:
            mess = MESS()
            mess.process(dem, poly)
            self.mess += mess

        # * Create NavMesh
        self.navmesh = NavMesh(self.mess)
        self.navmesh.set_points(self.position, self.goal)
        self.goal_in_sight = self.navmesh.goal_idx is not None

        logger.info("Running A*")
        path = PathPlanner.path_finder(self.navmesh, algorithm="astar")
        logger.success("A* finished")

        # * Move
        self.previous_position = self.position
        for pos in path:
            self.position = pos
            self.path = np.append(self.path, [pos], axis=0)
            poly = self.__calc_poly()
            if self.__need_observation(poly):
                break

        if self.goal_in_sight:
            return Status.REACHED

        return Status.SUCCESS

    def get_path(self):
        _, idx = np.unique(self.path[:, [0, 1]], axis=0, return_index=True)
        path = self.path[np.sort(idx)].T

        tck, u = interpolate.splprep(path)
        path = interpolate.splev(u, tck)
        path = np.array(path).T
        return path

    def get_poly(self):
        return self.__calc_poly()

    def __observe(self, poly):
        logger.info("Calculating current observation")
        extent = Conversion.poly_to_extent(poly)
        position = (self.position[0] - extent[2], self.position[1] - extent[0])
        dem = self.world[extent[2] : extent[3], extent[0] : extent[1]]
        return self.transform(position, dem)

    def __need_observation(self, poly):
        if self.mess is not None:
            unknown_area = poly.difference(self.mess.polygon).area
            ratio = unknown_area / (self.local_size ** 2)
            if ratio < self.recalculate_threshold:
                return False
        return True

    def __calc_poly(self):
        c_x = int(math.ceil(self.position[0]))
        c_y = int(math.ceil(self.position[1]))

        c_extent = [
            min(c_x - self.local_size // 2, c_x + self.local_size // 2),
            max(c_x - self.local_size // 2, c_x + self.local_size // 2),
            min(c_y - self.local_size // 2, c_y + self.local_size // 2),
            max(c_y - self.local_size // 2, c_y + self.local_size // 2),
        ]

        if self.agent_extent_bias == "inverse_diagonal":
            # Calculate direction
            p_x = int(math.ceil(self.previous_position[0]))
            p_y = int(math.ceil(self.previous_position[1]))
            a = math.atan2(c_y - p_y, c_x - p_x)
            p = 22.5 * math.pi / 180.0

            if c_x == p_x and c_y == p_y:  # center
                pass
            elif -p < a < p:  # right
                c_extent[0] += self.local_size // 4
                c_extent[1] += self.local_size // 4
            elif p < a < 3 * p:  # upper right
                c_extent[0] += self.local_size // 4
                c_extent[1] += self.local_size // 4
                c_extent[2] += self.local_size // 4
                c_extent[3] += self.local_size // 4
            elif 3 * p < a < 5 * p:  # upper
                c_extent[2] += self.local_size // 4
                c_extent[3] += self.local_size // 4
            elif 5 * p < a < 7 * p:  # upper left
                c_extent[0] -= self.local_size // 4
                c_extent[1] -= self.local_size // 4
                c_extent[2] += self.local_size // 4
                c_extent[3] += self.local_size // 4
            elif 7 * p < a or -7 * p > a:  # left
                c_extent[0] -= self.local_size // 4
                c_extent[1] -= self.local_size // 4
            elif -5 * p > a:  # lower left
                c_extent[0] -= self.local_size // 4
                c_extent[1] -= self.local_size // 4
                c_extent[2] -= self.local_size // 4
                c_extent[3] -= self.local_size // 4
            elif -3 * p > a:  # lower
                c_extent[2] -= self.local_size // 4
                c_extent[3] -= self.local_size // 4
            elif -p > a:  # lower right
                c_extent[0] += self.local_size // 4
                c_extent[1] += self.local_size // 4
                c_extent[2] -= self.local_size // 4
                c_extent[3] -= self.local_size // 4

        c_poly = Conversion.extent_to_poly(c_extent)
        c_poly = c_poly.intersection(self.world_extent)
        return c_poly