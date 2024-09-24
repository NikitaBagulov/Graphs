from graph import Graph
from graph_visualizer import GraphVisualizer
from node import Node
from dijkstra import Dijkstra

node1 = Node("1")
node2 = Node("2")
node3 = Node("3")
node4 = Node("4")
node5 = Node("5")
node6 = Node("6")
node7 = Node("7")
node8 = Node("8")
node9 = Node("9")

graph = Graph()
graph.add_node(node1)
graph.add_node(node2)
graph.add_node(node3)
graph.add_node(node4)
graph.add_node(node5)
graph.add_node(node6)
graph.add_node(node7)
graph.add_node(node8)
graph.add_node(node9)

graph.add_edge(node1, node2, 10)
graph.add_edge(node1, node3, 6)
graph.add_edge(node1, node4, 8)
graph.add_edge(node2, node4, 5)
graph.add_edge(node2, node5, 13)
graph.add_edge(node2, node7, 11)
graph.add_edge(node3, node5, 3)
graph.add_edge(node4, node3, 2)
graph.add_edge(node4, node5, 5)
graph.add_edge(node4, node6, 7)
graph.add_edge(node4, node7, 12)
graph.add_edge(node5, node6, 9)
graph.add_edge(node5, node9, 12)
graph.add_edge(node6, node8, 8)
graph.add_edge(node6, node9, 10)
graph.add_edge(node7, node6, 4)
graph.add_edge(node7, node8, 6)
graph.add_edge(node7, node9, 16)
graph.add_edge(node8, node9, 15)

# Запускаем алгоритм Дейкстры
dijkstra = Dijkstra(graph)
shortest_path, distance = dijkstra.shortest_path(node1, node9)

print("Кратчайший путь:", [node.name for node in shortest_path])
print("Длина пути:", distance)

visualizer = GraphVisualizer(graph)
visualizer.animate_path(shortest_path)
