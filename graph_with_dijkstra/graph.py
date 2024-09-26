from node import Node

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
            end.add_neighbour(begin, weight)

    def remove_edge(self, begin:Node, end:Node):
        if begin in self.nodes and end in self.nodes:
            begin.remove_neighbour(end)
            end.remove_neighbour(begin)

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
    
    def load_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            next(f)
            for line in f:
                source, target, weight = line.strip().split()
                source_node = self.get_or_create_node(source)
                target_node = self.get_or_create_node(target)
                self.add_edge(source_node, target_node, int(weight))

    def get_or_create_node(self, node_name: str) -> Node:
        for node in self.nodes:
            if node.get_name() == node_name:
                return node
        new_node = Node(node_name)
        self.add_node(new_node)
        return new_node

    

