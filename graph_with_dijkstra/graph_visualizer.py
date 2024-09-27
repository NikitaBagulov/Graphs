import matplotlib.pyplot as plt
import numpy as np
from graph import Graph

class GraphVisualizer:
    def __init__(self, graph: Graph):
        self.graph = graph

    def draw_graph(self, start=None, end=None, path=None):
        plt.clf()
        pos = {}
        num_nodes = len(self.graph.nodes)

        for i, node in enumerate(self.graph.nodes):
            angle = 2 * np.pi * i / num_nodes
            pos[node.get_name()] = (np.cos(angle), np.sin(angle))

        for node_name, (x, y) in pos.items():
            if start and node_name == start.get_name():
                color = 'lightgreen'  
            elif end and node_name == end.get_name():
                color = 'salmon'
            else:
                color = 'lightblue'
            plt.scatter(x, y, s=2000, color=color, edgecolors='black')
            plt.text(x, y, node_name, fontsize=12, ha='center', va='center')

        for node in self.graph:
            for neighbour, weight in node.nb_begin():
                x_start, y_start = pos[node.get_name()]
                x_end, y_end = pos[neighbour.get_name()]
                plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                          head_width=0.05, head_length=0.1, fc='gray', ec='gray')
                mid_x = (x_start + x_end) / 2
                mid_y = (y_start + y_end) / 2
                plt.text(mid_x, mid_y, str(weight), fontsize=10, ha='center', va='center')

        if path:
            for i in range(len(path) - 1):
                x_start, y_start = pos[path[i].get_name()]
                x_end, y_end = pos[path[i + 1].get_name()]
                plt.plot([x_start, x_end], [y_start, y_end], color='red', linewidth=2, linestyle='dashed')

        plt.title("Визуализация графа с направленными рёбрами")
        plt.axis('equal')
        plt.grid(False)
        plt.draw()

    def animate_path(self, path):
        plt.ion()
        plt.figure(figsize=(8, 6))
        for i in range(len(path) - 1):
            self.draw_graph(start=path[0], end=path[-1], path=path[:i+2])
            plt.pause(1)
        plt.ioff()
        plt.show()
