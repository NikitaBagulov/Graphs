from graph import Graph, Node
from graph_visualizer import GraphVisualizer
from dijkstra import Dijkstra

graph = Graph()
graph.load_from_file("graph_with_dijkstra/1000.txt")
start = graph.get_or_create_node('0')
target = graph.get_or_create_node('999')
dijkstra = Dijkstra(graph)
shortest_path, distance = dijkstra.shortest_path(start, target)
print("Кратчайший путь:", [node.name for node in shortest_path])
print("Длина пути:", distance)
# visualizer = GraphVisualizer(graph)
# visualizer.animate_path(shortest_path)
