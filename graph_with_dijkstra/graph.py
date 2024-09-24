from node import Node
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self) -> None:
        self.nodes = set()

    def add_node(self, node):
        self.nodes.add(node)

    def remove_node(self, node: Node):
        self.nodes.remove(node)
        for neighbour in list(node.neighbours.keys()):
            node.remove_neighbour(neighbour)

    def add_edge(self, begin:Node, end:Node, weight: float):
        if begin in self.nodes and end in self.nodes:
            begin.add_neighbour(end, weight)

    def remove_edge(self, begin:Node, end:Node):
        if begin in self.nodes and end in self.nodes:
            begin.remove_neighbour(end)

    def __iter__(self):
        return iter(self.nodes)
    
    def __repr__(self) -> str:
        return f"Graph({', '.join([node.get_name() for node in self.nodes])})"
    
    def display_edges(self):
        edges = []
        for node in self.nodes:
            for neighbour, weight in node.nb_begin():
                edges.append((node.get_name(), neighbour.get_name(), weight))
        return edges
    
    # def draw_graph(self):
    #     plt.figure(figsize=(8, 6))
    #     pos = {}
    #     num_nodes = len(self.nodes)
    #     for i, node in enumerate(self.nodes):
    #         angle = 2 * np.pi * i / num_nodes
    #         pos[node.get_name()] = (np.cos(angle), np.sin(angle))

    #     for node_name, (x, y) in pos.items():
    #         plt.scatter(x, y, s=2000, color='lightblue', edgecolors='black')
    #         plt.text(x, y, node_name, fontsize=12, ha='center', va='center')

    #     for node in self:
    #         for neighbour, weight in node.nb_begin():
    #             x_start, y_start = pos[node.get_name()]
    #             x_end, y_end = pos[neighbour.get_name()]
    #             plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
    #                     head_width=0.05, head_length=0.1, fc='gray', ec='gray')
    #             mid_x = (x_start + x_end) / 2
    #             mid_y = (y_start + y_end) / 2
    #             plt.text(mid_x, mid_y, str(weight), fontsize=10, ha='center', va='center')

    #     plt.title("Визуализация графа с направленными рёбрами")
    #     plt.axis('equal')
    #     plt.grid(False)
    #     plt.show()

    

