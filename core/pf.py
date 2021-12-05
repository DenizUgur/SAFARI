import heapq
import numpy as np
from collections import deque

from .mess import NavMesh


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def TopKey(self):
        return heapq.nsmallest(1, self.elements)[0][0]

    def delete(self, item):
        self.elements = [e for e in self.elements if e[1] != item]
        heapq.heapify(self.elements)

    def __iter__(self):
        for _, item in self.elements:
            yield item


class PathPlanner:
    @staticmethod
    def reconstruct_path(mesh, came_from, start, goal):
        current = came_from[goal]
        path = []
        while current != start:
            path.append(mesh(current))
            current = came_from[current]
        path.append(mesh(start))
        path.reverse()
        return np.array(path)

    @staticmethod
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

        return came_from

    @staticmethod
    def astar_search(mesh, start, goal):
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
                    priority = new_cost + mesh.h(next, goal)
                    frontier.put(next, priority)
                    came_from[next] = current

        return came_from

    @staticmethod
    def path_finder(NM: NavMesh, algorithm="dijkstra"):
        method = getattr(PathPlanner, f"{algorithm}_search")

        came_from = method(NM, NM.start_idx, NM.goal_idx or NM.goal_nearest_idx)

        path = PathPlanner.reconstruct_path(
            NM, came_from, NM.start_idx, NM.goal_idx or NM.goal_nearest_idx
        )

        return path


class ADStar:
    def __init__(self, navmesh: NavMesh, eps=100.0):
        self.eps = eps
        self.navmesh = navmesh
        self.start = self.navmesh.start_idx

        self.g, self.rhs, self.OPEN = {}, {}, {}

        for vertex, __ in enumerate(self.navmesh.vertices):
            self.rhs[vertex] = float("inf")
            self.g[vertex] = float("inf")

        self.rhs[NavMesh.GOAL] = 0.0
        self.g[NavMesh.GOAL] = float("inf")
        self.OPEN[NavMesh.GOAL] = self.Key(NavMesh.GOAL)

        self.CLOSED, self.INCONS = set(), dict()
        self.VISITED = set()

    def Run(self):
        self.ComputeOrImprovePath()
        self.VISITED = set()

        while True:
            if self.eps <= 1.0:
                break
            self.eps -= 0.5

            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.VISITED = set()

    def Update(self):
        for vertex in self.navmesh.new_vertices:
            self.g[vertex] = float("inf")
            self.rhs[vertex] = float("inf")

        for vertex in self.navmesh.new_vertices:
            for sn in self.navmesh.neighbors(vertex):
                self.UpdateState(sn)

        self.UpdateState(NavMesh.GOAL)

        self.eps += 2.0
        while True:
            if len(self.INCONS) == 0:
                break

            self.OPEN.update(self.INCONS)
            for s in self.OPEN:
                self.OPEN[s] = self.Key(s)
            self.CLOSED = set()
            self.ComputeOrImprovePath()
            self.VISITED = set()

            if self.eps <= 1.0:
                break

    def ComputeOrImprovePath(self):
        while True:
            s, v = self.TopKey()
            if v >= self.Key(self.start) and self.rhs[self.start] == self.g[self.start]:
                break

            self.OPEN.pop(s)
            self.VISITED.add(s)

            if self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                self.CLOSED.add(s)
                for sn in self.navmesh.neighbors(s):
                    self.UpdateState(sn)
            else:
                self.g[s] = float("inf")
                for sn in self.navmesh.neighbors(s):
                    self.UpdateState(sn)
                self.UpdateState(s)

    def UpdateState(self, s):
        if s != NavMesh.GOAL:
            self.rhs[s] = float("inf")
            for x in self.navmesh.neighbors(s):
                self.rhs[s] = min(self.rhs[s], self.g[x] + self.navmesh.cost(s, x))
        if s in self.OPEN:
            self.OPEN.pop(s)

        if self.g[s] != self.rhs[s]:
            if s not in self.CLOSED:
                self.OPEN[s] = self.Key(s)
            else:
                self.INCONS[s] = 0

    def Key(self, s):
        if self.g[s] > self.rhs[s]:
            return [
                self.rhs[s] + self.eps * self.navmesh.h(self.start, s),
                self.rhs[s],
            ]
        else:
            return [self.g[s] + self.navmesh.h(self.start, s), self.g[s]]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.OPEN, key=self.OPEN.get)
        return s, self.OPEN[s]

    def ExtractPath(self):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.navmesh(self.start)]
        s = self.start

        while True:
            g_list = {}
            for x in self.navmesh.neighbors(s):
                g_list[x] = self.g[x]
            s = min(g_list, key=g_list.get)
            path.append(self.navmesh(s))
            if s == NavMesh.GOAL:
                break

        return np.array(path)


class DStarLite:
    def __init__(self, navmesh: NavMesh) -> None:
        self.navmesh = navmesh
        self.position = self.navmesh.start_idx
        self.last_position = self.position
        self.path = []

        self.U = PriorityQueue()
        self.Km = 0
        self.g, self.rhs = {}, {}

        for vertex, __ in enumerate(self.navmesh.vertices):
            self.rhs[vertex] = float("inf")
            self.g[vertex] = float("inf")

        self.g[NavMesh.GOAL] = float("inf")
        self.rhs[NavMesh.GOAL] = 0
        self.U.put(NavMesh.GOAL, self.CalculateKey(NavMesh.GOAL))

    def CalculateKey(self, s):
        return [
            min(self.g[s], self.rhs[s]) + self.navmesh.h(self.position, s) + self.Km,
            min(self.g[s], self.rhs[s]),
        ]

    def UpdateVertex(self, u):
        if u != NavMesh.GOAL:
            for s in self.navmesh.neighbors(u):
                self.rhs[u] = min(self.rhs[u], self.navmesh.cost(u, s) + self.g[s])

        self.U.delete(u)
        if self.g[u] != self.rhs[u]:
            self.U.put(u, self.CalculateKey(u))

    def ComputeShortestPath(self):
        last_nodes = deque(maxlen=10)
        while (
            self.U.TopKey() < self.CalculateKey(self.position)
            or self.rhs[self.position] != self.g[self.position]
        ):
            Kold = self.U.TopKey()
            u = self.U.get()

            # Check loop status
            last_nodes.append(u)
            if len(last_nodes) == 10 and len(set(last_nodes)) < 3:
                raise Exception("Fail! Stuck in a loop")

            if Kold < self.CalculateKey(u):
                self.U.put(u, self.CalculateKey(u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.navmesh.neighbors(u):
                    self.UpdateVertex(s)
            else:
                self.g[u] = float("inf")
                for s in self.navmesh.neighbors(u):
                    self.UpdateVertex(s)
                self.UpdateVertex(u)

    def Run(self):
        self.ComputeShortestPath()
        self.path.append(self.position)
        self.Update(newdata=False)

    def Update(self, newdata=True):
        if newdata:
            for vertex in self.navmesh.new_vertices:
                self.g[vertex] = float("inf")
                self.rhs[vertex] = float("inf")

        while self.position != NavMesh.GOAL:
            if self.g[self.position] == float("inf"):
                raise Exception("Fail! No path available")

            sm, costm = -1, float("inf")
            for s in self.navmesh.neighbors(self.position):
                cost = self.navmesh.cost(self.position, s) + self.g[s]
                if cost < costm:
                    costm = cost
                    sm = s
            self.position = sm
            self.path.append(self.position)

            if newdata:
                newdata = False
                self.Km = self.Km + self.navmesh.h(self.last_position, self.position)
                self.last_position = self.position

                for vertex in self.navmesh.new_vertices:
                    for sn in self.navmesh.neighbors(vertex):
                        self.UpdateVertex(sn)

                self.ComputeShortestPath()
            self.path.append(self.position)

    def ExtractPath(self):
        path = []
        self.path = self.path[:-3]
        for p in self.path:
            path.append(self.navmesh(p))

        self.position = self.path[-1]
        return np.array(path)
