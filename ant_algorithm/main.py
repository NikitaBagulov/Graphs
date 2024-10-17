from ant import AntAlgorithm
from graph import Graph
from node import Node

graph = Graph()

node_a = Node("a")
node_b = Node("b")
node_c = Node("c")
node_d = Node("d")
node_g = Node("g")
node_f = Node("f")

graph.add_node(node_a)
graph.add_node(node_b)
graph.add_node(node_c)
graph.add_node(node_d)
graph.add_node(node_f)
graph.add_node(node_g)

graph.add_edge(node_a, node_b, 3)
graph.add_edge(node_a, node_f, 1)
graph.add_edge(node_b, node_a, 3)
graph.add_edge(node_b, node_c, 8)
graph.add_edge(node_b, node_g, 3)
graph.add_edge(node_c, node_b, 3)
graph.add_edge(node_c, node_d, 1)
graph.add_edge(node_c, node_g, 1)
graph.add_edge(node_d, node_c, 8)
graph.add_edge(node_d, node_f, 1)
graph.add_edge(node_f, node_a, 3)
graph.add_edge(node_f, node_d, 3)
graph.add_edge(node_g, node_a, 3)
graph.add_edge(node_g, node_b, 3)
graph.add_edge(node_g, node_c, 3)
graph.add_edge(node_g, node_d, 5)
graph.add_edge(node_g, node_f, 4)

ant = AntAlgorithm(graph=graph,
                 num_ants=1,
                 num_iterations=100,
                 decay=0.1,
                 alpha=2,
                 beta=1)

best_cycle, best_distance = ant.run()
print("Best Cycle:", [node.get_name() for node in best_cycle])
print("Best Distance:", best_distance)