import os
import torch
import pymesh
import numpy as np
import urllib.request
from numba import cuda
from loguru import logger
from math import log10, sin, cos, atan2, ceil, hypot, atan

from model.aotgan import InpaintGenerator

TARGET_LEN = 2.0
RESOLUTION = 0.1


@cuda.jit(device=True)
def _cost_log(sx, sy, sz, ex, ey, ez, params):
    """
    https://www.desmos.com/calculator/ujo1cmktnm
    """
    RoR = (ez - sz) / hypot(sx - ex, sy - ey)
    a = atan(RoR)

    if a > 0:  # Uphill
        q0, q1 = params[0], params[1]
    else:  # Downhill
        q0, q1 = params[2], params[3]

    return max(min(log10(q0 * abs(a) + 1) ** q1, 1), 0)


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
                fz = terrain[fx, fy]
                lx = int(fe + tx)
                ly = int(se + ty)
                lz = terrain[lx, ly]
                cum_cost += _cost_log(fx, fy, fz, lx, ly, lz, params)

            costs[x, y] = cum_cost
        else:
            costs[x, y] = np.nan


def mesh_to_extent(vertices, extent):
    vertices[:, 0] *= RESOLUTION
    vertices[:, 0] += extent[0]
    vertices[:, 1] *= RESOLUTION
    vertices[:, 1] += extent[1]
    return vertices


def mesh_from_extent(vertices, extent):
    vertices[:, 0] -= extent[0]
    vertices[:, 0] /= RESOLUTION
    vertices[:, 1] -= extent[1]
    vertices[:, 1] /= RESOLUTION
    return vertices


class MESS:
    SAMPLE_SIZE = 96

    def __init__(self, cost_coeffs, local_size, wheels=None) -> None:
        # * Cost Related
        self.coeffs = cost_coeffs
        self.wheels = wheels
        if wheels is None:
            self.wheels = np.array([[-0.6, 0.6], [0.6, 0.6], [0.6, -0.6], [-0.6, -0.6]])

        # * AI Related
        logger.info("Initializing InpaintGenerator")

        if not os.path.exists("model.pt"):
            logger.info("Model does not exists, downloading now")
            urllib.request.urlretrieve(
                "https://github.com/DenizUgur/SAFARI/releases/download/V1/model.pt",
                "model.pt",
            )
            logger.success("Model downloaded")

        self.net = InpaintGenerator()
        self.net.load_state_dict(torch.load("model.pt"))
        self.net.cuda()
        self.net.eval()
        self.net_constant = 4.999878  # 39.9697265625
        self.block_dimension = 5
        self.block_dem_buffer = torch.zeros(
            1, self.block_dimension, 1, MESS.SAMPLE_SIZE, MESS.SAMPLE_SIZE
        ).cuda()
        self.block_mask_buffer = torch.zeros(
            1, self.block_dimension, 1, MESS.SAMPLE_SIZE, MESS.SAMPLE_SIZE
        ).cuda()
        logger.success("Initialized InpaintGenerator")

        # * Map related
        self.map_L1 = None  # 1x1
        self.map_L1_ref = None  # 1x1 DEM
        self.extent_L1 = None  # 1x1

        self.map_L2 = None  # 3x3
        self.map_L2_ref = None  # 3x3 DEM
        self.extent_L2 = None  # 3x3

        # * Preparation
        MESS.SAMPLE_SIZE = local_size
        xs, ys = np.meshgrid(range(local_size), range(local_size), indexing="xy")
        self.vertices = np.dstack([xs, ys]).reshape(-1, 2)

        faces = []
        logger.info("Initializing template faces")
        for i, aa in enumerate(self.vertices):
            x, y = aa
            if x < local_size - 1 and y < local_size - 1:
                faces.extend(
                    [
                        [i, i + 1, int(x + (local_size * (y + 1)))],
                        [
                            int(x + (local_size * (y + 1))),
                            i + 1,
                            int(x + (local_size * (y + 1))) + 1,
                        ],
                    ]
                )
        self.faces = np.array(faces)
        logger.success("Initialized template faces")

    def fix_mesh(self, mesh):
        global TARGET_LEN
        count = 0
        mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
        mesh, __ = pymesh.split_long_edges(mesh, TARGET_LEN)
        num_vertices = mesh.num_vertices
        while True:
            mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
            mesh, __ = pymesh.collapse_short_edges(
                mesh, TARGET_LEN, preserve_feature=True
            )
            mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
            if mesh.num_vertices == num_vertices:
                break

            count += 1
            if count > 10:
                break

        mesh = pymesh.resolve_self_intersection(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh = pymesh.compute_outer_hull(mesh)
        mesh, __ = pymesh.remove_duplicated_faces(mesh)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
        mesh, __ = pymesh.remove_isolated_vertices(mesh)

        return mesh

    def cube_former(self, dem, mask):
        if torch.all(self.block_dem_buffer == 0):
            self.block_dem_buffer[0, :, 0] = dem
            self.block_mask_buffer[0, :, 0] = mask
            return

        self.block_dem_buffer = torch.cat(
            (self.block_dem_buffer[:, 1:, :, :, :], dem.reshape(1, 1, 1, *dem.shape)),
            axis=1,
        )
        self.block_mask_buffer = torch.cat(
            (
                self.block_mask_buffer[:, 1:, :, :, :],
                mask.reshape(1, 1, 1, *mask.shape),
            ),
            axis=1,
        )

    def void_filler(self, dem):
        dem_min = np.nanmin(dem)
        nans = np.isnan(dem)

        if dem_min < 0:
            dem += abs(dem_min)
        else:
            dem -= dem_min

        dem[nans] = 0
        dem /= self.net_constant

        target = torch.from_numpy(dem.copy()).float().cuda()
        mask = torch.from_numpy(nans.astype(np.uint8)).float().cuda()
        self.cube_former(target, mask)

        with torch.no_grad():
            masked_tensor = (
                self.block_dem_buffer * (1 - self.block_mask_buffer).float()
            ) + self.block_mask_buffer
            pred_tensor = self.net(masked_tensor, self.block_mask_buffer)
            comp_tensor = (
                pred_tensor * self.block_mask_buffer
                + self.block_dem_buffer * (1 - self.block_mask_buffer)
            )

        result = comp_tensor[0, -1, 0].cpu().numpy()

        if dem_min < 0:
            result -= abs(dem_min)
        else:
            result += dem_min

        return result * self.net_constant

    def update_local(self, dem, extent):
        global RESOLUTION
        """
        dem -> 2D elevation information
        extent -> (xmin, ymin, xmax, ymax)
        """
        # * Create navigable mesh from given DEM
        logger.info("Running InpaintGenerator on L1")
        self.map_L1_ref = self.void_filler(dem)

        logger.info("Simplifying input DEM")
        referance = self.map_L1_ref.flatten().reshape(-1, 1)

        vertices = np.hstack([self.vertices, referance])
        mesh = pymesh.form_mesh(vertices, self.faces)
        mesh = self.fix_mesh(mesh)

        # * Extract edges from mesh
        mesh.enable_connectivity()

        edges = []
        for idx, _ in enumerate(mesh.vertices):
            edges.append(mesh.get_vertex_adjacent_vertices(idx))
        pad = len(max(edges, key=len))
        edges = np.array([list(i) + [-1] * (pad - len(i)) for i in edges])

        # * Update mesh according to extent
        self.extent_L1 = extent
        RESOLUTION = (extent[2] - extent[0]) / dem.shape[0]
        vertices = mesh_to_extent(np.copy(mesh.vertices), extent)

        self.map_L1 = (vertices, mesh.faces, edges)
        logger.success("L1 mesh generated")

    def update_L2_ref(self, dem, extent):
        self.map_L2_ref = dem
        self.extent_L2 = extent

    def update_costs(self):
        logger.info("Preparing for cost calculation on L2")
        orig_vertices, faces, edges, _ = self.map_L2

        # * Convert mesh from extent
        vertices = mesh_from_extent(np.copy(orig_vertices), self.extent_L2)

        pad = len(max(edges, key=len))
        costs = np.zeros_like(edges, dtype=float)

        threadsperblock = (pad, pad)
        blockspergrid_x = ceil(costs.shape[0] / threadsperblock[0])
        blockspergrid_y = ceil(costs.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        c_vertices = cuda.to_device(vertices)
        c_edges = cuda.to_device(edges)
        c_costs = cuda.to_device(costs)
        c_referance = cuda.to_device(np.ascontiguousarray(self.map_L2_ref))
        c_wheels = cuda.to_device(self.wheels)
        c_params = cuda.to_device(self.coeffs)

        logger.info("Starting calculation on the GPU")
        with cuda.defer_cleanup():
            cost[blockspergrid, threadsperblock](
                c_vertices, c_edges, c_costs, c_referance, c_wheels, c_params
            )

        h_costs = c_costs.copy_to_host()
        assert np.count_nonzero(edges != np.nan) == np.count_nonzero(
            h_costs != np.nan
        ), "Sanity Check: Calculated costs do not match edge count"

        self.map_L2 = (orig_vertices, faces, edges, h_costs)
        logger.success("Cost calculation completed")

    def process(self, no_merge=False):
        if self.map_L2 is None or no_merge:
            L1_v, L1_f, L1_e = self.map_L1
            self.map_L2 = np.copy(L1_v), np.copy(L1_f), np.copy(L1_e), None
            self.update_costs()
            return

        # * Get and merge L1 and L2
        L1_v, L1_f, _ = self.map_L1
        L2_v, L2_f, _, _ = self.map_L2

        merged = pymesh.merge_meshes(
            [pymesh.form_mesh(L1_v, L1_f), pymesh.form_mesh(L2_v, L2_f)]
        )
        M_v, M_f = merged.vertices, merged.faces

        # * Discard vertices outside L2 extent
        L2_xmin, L2_ymin, L2_xmax, L2_ymax = self.extent_L2

        mask = M_v[:, 0] >= L2_xmin
        mask &= M_v[:, 0] <= L2_xmax
        mask &= M_v[:, 1] >= L2_ymin
        mask &= M_v[:, 1] <= L2_ymax

        merged.enable_connectivity()
        if np.count_nonzero(~mask) > 0:
            new_faces = np.copy(M_f)
            rm_faces = []
            for i_vertex in np.argwhere(~mask):
                rm_faces.extend(merged.get_vertex_adjacent_faces(i_vertex[0]))
            new_faces = np.delete(new_faces, np.unique(rm_faces), 0)

            merged = pymesh.form_mesh(M_v[mask], new_faces)
            merged, _ = pymesh.remove_isolated_vertices(merged)
            merged.enable_connectivity()

        edges = []
        for idx, _ in enumerate(merged.vertices):
            edges.append(merged.get_vertex_adjacent_vertices(idx))
        pad = len(max(edges, key=len))
        edges = np.array([list(i) + [-1] * (pad - len(i)) for i in edges])

        self.map_L2 = merged.vertices, merged.faces, edges, None
        self.update_costs()