import numba
import pymesh
import numpy as np
from scipy import spatial
from loguru import logger
from scipy import interpolate
from numba import cuda, jit, njit
from shapely.geometry import Polygon
from math import log10, sin, cos, atan2, ceil, hypot, atan

TARGET_LEN = 2.0


@cuda.jit(device=True)
def _cost_log(sx, sy, sz, ex, ey, ez, params):
    """
    https://www.desmos.com/calculator/kjgb6ak4qs
    """
    t = ez - sz
    b = hypot(sx - ex, sy - ey)
    if t == 0 or b == 0:
        return 0

    RoR = t / b
    a = atan(RoR)

    if abs(a) > 0.523599:  # 30 deg
        return np.inf

    if a > 0:  # Uphill
        q0, q1 = params[0], params[1]
    else:  # Downhill
        q0, q1 = params[2], params[3]

    return max(log10(q0 * abs(a) + 1) ** q1, 0)


@cuda.jit
def cost(vertices, edges, costs, terrain, wheels, params):
    """
    wheels -> FL, FR, RR, RL
    """
    x, y = cuda.grid(2)
    if x < costs.shape[0] and y < costs.shape[1]:

        target_index = edges[x, y]
        if target_index != -1:
            cx, cy, cz = vertices[x]  # Current Node
            tx, ty, tz = vertices[target_index]  # Target Node
            a = atan2(ty - cy, tx - cx)
            c_a = cos(a)
            s_a = sin(a)

            cum_cost = _cost_log(cx, cy, cz, tx, ty, tz, params)
            for (wx, wy) in wheels:
                fe = wx * c_a - wy * s_a
                se = wx * s_a + wy * c_a
                fx = int(fe + cx)
                fy = int(se + cy)
                fz = terrain[fy, fx]
                lx = int(fe + tx)
                ly = int(se + ty)
                lz = terrain[ly, lx]
                cum_cost += _cost_log(fx, fy, fz, lx, ly, lz, params)

            costs[x, y] = cum_cost
        else:
            costs[x, y] = -1


@jit(nopython=True)
def pointinpolygon(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i, 0], points[i, 1], polygon)
    return D


class Conversion:
    @staticmethod
    def poly_to_extent(poly):
        bounds = poly.bounds
        return list(map(int, [bounds[0], bounds[2], bounds[1], bounds[3]]))

    @staticmethod
    def extent_to_poly(extent):
        return Polygon(
            [
                (extent[0], extent[2]),
                (extent[1], extent[2]),
                (extent[1], extent[3]),
                (extent[0], extent[3]),
            ]
        )


class MESS:
    VERTICES = None
    FACES = None

    def __init__(self, coeffs=None, wheels=None) -> None:
        """
        cost_coeffs -> [q0, q1, q2, q3]
        """
        # * Cost Related
        self.coeffs = coeffs
        if coeffs is None:
            self.coeffs = np.array([11.09, 0.49, 9.72, 0.62])

        self.wheels = wheels
        if wheels is None:
            self.wheels = np.array([[-0.6, 0.6], [0.6, 0.6], [0.6, -0.6], [-0.6, -0.6]])

        # * Map related
        self.map = None  # 1x1
        self.map_ref = None  # 1x1 DEM
        self.extent = None
        self.polygon = None
        self.invalid_vertices = None

        # * Map internals
        self.max_neighbors = 20
        self.vertices = None
        self.edges = []
        self.new_vertices = None

    def __prepare(self):
        # * Preparation
        xmax = self.extent[1] - self.extent[0]
        ymax = self.extent[3] - self.extent[2]

        self.mesh_target_len = 2.0
        xs, ys = np.meshgrid(range(xmax), range(ymax), indexing="xy")
        self.VERTICES = np.dstack([xs, ys]).reshape(-1, 2)

        faces = []
        logger.info("Initializing template faces")
        for i, aa in enumerate(self.VERTICES):
            x, y = aa
            if x < xmax - 1 and y < ymax - 1:
                faces.extend(
                    [
                        [i, i + 1, int(x + (xmax * (y + 1)))],
                        [
                            int(x + (xmax * (y + 1))),
                            i + 1,
                            int(x + (xmax * (y + 1))) + 1,
                        ],
                    ]
                )
        self.FACES = np.array(faces)
        logger.success("Initialized template faces")

    def __fix_mesh(self, mesh, improvement_thres=0.8):
        mesh, __ = pymesh.split_long_edges(mesh, self.mesh_target_len)
        num_vertices = len(mesh.vertices)

        for __ in range(10):
            mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
            mesh, __ = pymesh.collapse_short_edges(
                mesh, self.mesh_target_len, preserve_feature=True
            )
            mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)

            if len(mesh.vertices) < num_vertices * improvement_thres:
                break

        mesh = pymesh.resolve_self_intersection(mesh)
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
        mesh, __ = pymesh.remove_isolated_vertices(mesh)

        return mesh

    def __update_local(self):
        # * Interpolate NaNs and mask out NaNs from mesh after __fix_mesh
        logger.info("Interpolating unnavigable areas")

        x = np.arange(0, self.map.shape[1])
        y = np.arange(0, self.map.shape[0])
        self.map = np.ma.masked_invalid(self.map)
        xx, yy = np.meshgrid(x, y)

        x1 = xx[~self.map.mask]
        y1 = yy[~self.map.mask]

        self.map_ref = interpolate.griddata(
            (x1, y1), self.map[~self.map.mask].ravel(), (xx, yy), method="nearest"
        )

        # * Create navigable mesh from given DEM
        logger.info("Simplifying input DEM")
        referance = self.map_ref.flatten().reshape(-1, 1)

        vertices = np.hstack([self.VERTICES, referance])
        mesh = pymesh.form_mesh(vertices, self.FACES)
        mesh = self.__fix_mesh(mesh)
        mesh.enable_connectivity()

        # * Calculate valid vertices
        logger.info("Filtering unnavigable areas")
        invalid_vertices = np.argwhere(self.map.mask)
        floored = np.floor(mesh.vertices).astype(int)[:, [1, 0]]
        ceiled = np.ceil(mesh.vertices).astype(int)[:, [1, 0]]

        search = (floored[:, None] == invalid_vertices).all(-1)
        search |= (ceiled[:, None] == invalid_vertices).all(-1)
        args = np.where(search.any(0), search.argmax(0), np.nan)
        self.invalid_vertices = args[~np.isnan(args)].astype(int)

        # * Extract edges from mesh
        for idx, _ in enumerate(mesh.vertices):
            if idx in self.invalid_vertices:
                self.edges.append([-1])
                continue

            neighbors = mesh.get_vertex_adjacent_vertices(idx)
            neighbors = filter(lambda x: x not in self.invalid_vertices, neighbors)
            self.edges.append(list(neighbors))
        self.edges = np.array(
            [list(i) + [-1] * (self.max_neighbors - len(i)) for i in self.edges]
        )

        self.vertices = np.copy(mesh.vertices)
        logger.success("Mesh generated")

    def __update_costs(self):
        logger.info("Preparing for cost calculation")
        costs = np.zeros_like(self.edges, dtype=float)

        threadsperblock = (self.max_neighbors, self.max_neighbors)
        blockspergrid_x = ceil(costs.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(costs.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        c_vertices = cuda.to_device(self.vertices)
        c_edges = cuda.to_device(self.edges)
        c_costs = cuda.to_device(costs)
        c_referance = cuda.to_device(np.ascontiguousarray(self.map_ref))
        c_wheels = cuda.to_device(self.wheels)
        c_params = cuda.to_device(self.coeffs)

        logger.info("Starting calculation on the GPU")
        with cuda.defer_cleanup():
            cost[blockspergrid, threadsperblock](
                c_vertices, c_edges, c_costs, c_referance, c_wheels, c_params
            )

        self.costs = c_costs.copy_to_host()
        logger.success("Cost calculation completed")

    def __add__(self, other):
        logger.info("Starting addition of two MESS instances")
        # * Start with vertices
        self.new_vertices_start = len(self.vertices)
        self.new_vertices = range(
            self.new_vertices_start, self.new_vertices_start + len(other.vertices)
        )
        self.vertices = np.concatenate((self.vertices, other.vertices), axis=0)

        # * Modify other edges
        other_edges = np.copy(other.edges)
        other_edges[other_edges != -1] += self.new_vertices_start
        self.edges = np.concatenate((self.edges, other_edges), axis=0)

        # * Now the costs
        self.costs = np.concatenate((self.costs, other.costs), axis=0)

        # * Finally connect a few edges between self and other
        intersect_region = self.polygon.intersection(other.polygon)
        I_r = np.array(intersect_region.exterior.coords)

        # * Find intersection vertices
        # self is A and other is B
        A_v = self.vertices[: self.new_vertices_start, [0, 1]]
        A_vi = parallelpointinpolygon(A_v, I_r)
        (A_vi,) = np.where(A_vi == True)

        B_v = self.vertices[self.new_vertices_start :, [0, 1]]
        B_vi = parallelpointinpolygon(B_v, I_r)
        (B_vi,) = np.where(B_vi == True)
        B_vi += self.new_vertices_start

        # * Just use A and connect each intersection vertex to B
        tree = spatial.KDTree(self.vertices[self.new_vertices_start :])
        for v in A_vi:
            vertex = self.vertices[v]
            _, c_idx = tree.query(vertex)

            # c_idx in B == v in A
            c_idx = c_idx + self.new_vertices_start
            c_vertex = self.vertices[c_idx]

            # Update edges
            (v_i,) = np.where(self.edges[v] == -1)
            self.edges[v][v_i[0]] = c_idx

            (c_i,) = np.where(self.edges[c_idx] == -1)
            self.edges[c_idx][c_i[0]] = v

            # Update costs
            (v_i,) = np.where(self.costs[v] == -1)
            self.costs[v][v_i[0]] = _cost_log(*vertex, *c_vertex, self.coeffs)

            (c_i,) = np.where(self.costs[c_idx] == -1)
            self.costs[c_idx][c_i[0]] = _cost_log(*c_vertex, *vertex, self.coeffs)

        self.polygon = self.polygon.union(other.polygon)
        logger.success(f"Done with {len(A_vi)} new edges")
        return self

    def process(self, dem, polygon):
        """
        dem -> 2D elevation information
        """
        self.map = dem
        self.map_ref = dem
        self.polygon = polygon
        self.extent = Conversion.poly_to_extent(polygon)

        self.__prepare()
        self.__update_local()
        self.__update_costs()

        self.vertices[:, 0] += self.extent[0]
        self.vertices[:, 1] += self.extent[2]

        return (self.vertices, self.edges, self.costs)


class NavMesh:
    GOAL = "goal"

    def __init__(self, mess: MESS) -> None:
        """
        Initialize with MESS.process() return value
        """
        self.start, self.start_idx = None, None
        self.goal, self.goal_idx = None, None

        self.update_from(mess)

    def __update_units(self):
        self.h_unit = 1
        h_unit = -np.inf
        for _from, _tos in enumerate(self.edges):
            for _to in _tos:
                if _to == -1:
                    continue
                h = self.h(_from, _to)
                if h > h_unit:
                    h_unit = h
        self.h_unit = h_unit

        self.c_unit = self.costs[self.costs != np.inf].max()

    def update_from(self, mess: MESS):
        self.vertices = mess.vertices
        self.edges = mess.edges
        self.costs = mess.costs
        self.new_vertices = mess.new_vertices
        self.__update_units()

    def set_points(self, start, goal=None):
        if goal is None and self.goal is None:
            raise Exception("No goal has been set")

        self.start = np.array(start)
        if goal:
            self.goal = np.array(goal)

        self.update_points(initial=True)

    def update_points(self, initial=False):
        tree = spatial.KDTree(self.vertices)

        if initial:
            __, self.start_idx = tree.query(self.start)

        if self.goal_idx is None:
            d, self.goal_nearest_idx = tree.query(self.goal)

            if d <= 4:
                self.goal_idx = self.goal_nearest_idx

    def neighbors(self, idx):
        n = []

        if self.goal_idx is None:
            if idx == NavMesh.GOAL:
                return [self.goal_nearest_idx]
            if idx == self.goal_nearest_idx:
                n.append(NavMesh.GOAL)
        else:
            if idx == NavMesh.GOAL:
                return [self.goal_idx]
            if idx == self.goal_idx:
                n.append(NavMesh.GOAL)

        for e in self.edges[idx]:
            if e != -1:
                n.append(e)
        return n

    def cost(self, _from, _to):
        if _from == NavMesh.GOAL or _to == NavMesh.GOAL:
            return self.c_unit

        _n = list(self.neighbors(_from))
        if NavMesh.GOAL in _n:
            _n.remove(NavMesh.GOAL)

        (_to,) = np.where(_n == _to)
        return self.costs[_from, _to] / self.c_unit

    def h(self, _from, _to):
        x1, y1, __ = self.vertices[_from]

        if _to == NavMesh.GOAL:
            x2, y2, __ = self.goal
        else:
            x2, y2, __ = self.vertices[_to]

        return hypot(x1 - x2, y1 - y2) / self.h_unit

    def __call__(self, idx):
        if idx == NavMesh.GOAL:
            if self.goal_idx is None:
                return self.goal
            else:
                idx = self.goal_idx

        return self.vertices[idx]
