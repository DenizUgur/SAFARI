import heapq
import numpy as np
from math import hypot

from .rdp import *


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class NavMesh:
    def __init__(self, vertices, edges, costs, valid_vertices) -> None:
        self.vertices = vertices
        self.edges = edges
        self.costs = costs
        self.valid_vertices, = np.where(valid_vertices)

        self.unit = -np.inf
        for _from, _tos in enumerate(self.edges):
            for _to in _tos:
                h = self.heuristic(_from, _to)
                if h > self.unit:
                    self.unit = h

    def neighbors(self, idx):
        _n = []
        for _e in self.edges[idx]:
            if _e == -1:
                continue
            if _e in self.valid_vertices:
                _n.append(_e)
        return _n

    def cost(self, _from, _to):
        _n = self.neighbors(_from)
        (_to,) = np.where(_n == _to)
        return self.costs[_from, _to]

    def heuristic(self, _from, _to):
        x1, y1, z1 = self.vertices[_from]
        x2, y2, z2 = self.vertices[_to]
        return hypot(x1 - x2, y1 - y2, z1 - z2)

    def __call__(self, idx):
        return self.vertices[idx]


def reconstruct_path(mesh, came_from, start, goal):
    current = list(came_from.keys())[-1]
    path = []
    while current != start:
        path.append(mesh(current))
        current = came_from[current]
    path.append(mesh(start))
    path.reverse()
    return np.array(path)


def dijkstra_search(mesh, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for i, next in enumerate(mesh.neighbors(current)):
            new_cost = cost_so_far[current] + mesh.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def a_star_search(mesh, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for i, next in enumerate(mesh.neighbors(current)):
            new_cost = cost_so_far[current] + mesh.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + mesh.heuristic(next, goal) / mesh.unit
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def path_finder(NM, si, ei):
    came_from, cost_so_far = dijkstra_search(NM, si, ei)
    path = reconstruct_path(NM, came_from, si, ei)

    # TODO: B-Spline is already being handled on path tracker
    # path = rdp(path, epsilon=1.0)
    # tck, u = interpolate.splprep([path[:, 0], path[:, 1], path[:, 2]])
    # path = interpolate.splev(u, tck)
    # path = np.array(path).T

    return path, cost_so_far