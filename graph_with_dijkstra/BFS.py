from collections import deque

class BFS:
    def __init__(self, graph):
        self.graph = graph

    def connected(self, begin, end):
        nodes = deque([begin])
        visited = set()

        while nodes:
            next_node = nodes.popleft()
            if next_node == end:
                return True
            visited.add(next_node)

            for neighbour in next_node.nb_begin():
                if neighbour not in visited:
                    nodes.append(neighbour)

        return False
