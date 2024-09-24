from graph import Graph
from node import Node

class DFS:
    def __init__(self, graph: Graph):
        self.graph: Graph = graph
        self.visited = set()

    def connected(self, begin: Node, end: Node):
        self.visited.clear()
        return self._connected(begin, end, 0)

    def _connected(self, begin:Node, end:Node, depth:int):
        if begin == end:
            return True
        self.visited.add(begin)

        for neighbour in begin.nb_begin():
            if neighbour not in self.visited:
                if self._connected(neighbour, end, depth + 1):
                    return True

        return False
