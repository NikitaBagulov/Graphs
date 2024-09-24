from graph import Graph
from node import Node
import heapq

class Dijkstra:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def shortest_path(self, start: Node, target: Node):
        distances = {node: float('inf') for node in self.graph.nodes}
        distances[start] = 0

        predecessors = {node: None for node in self.graph.nodes}
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_distance > distances[current_node]:
                continue

            if current_node == target:
                break

            for neighbour, weight in current_node.nb_begin():
                distance = current_distance + weight

                if distance < distances[neighbour]:
                    distances[neighbour] = distance
                    predecessors[neighbour] = current_node
                    heapq.heappush(priority_queue, (distance, neighbour))

        path = []
        current = target
        while current is not None:
            path.insert(0, current)
            current = predecessors[current]

        if path[0] != start:
            return None, float('inf') 
        return path, distances[target]
