
class Node:
    def __init__(self, name) -> None:
        self.name:str = name
        self.neighbours = {}

    def add_neighbour(self, neighbour, weight: float):
        self.neighbours[neighbour] = weight

    def remove_neighbour(self, neighbour):
        if neighbour in self.neighbours:
            del self.neighbours[neighbour]

    def get_name(self):
        return self.name
    
    def nb_begin(self):
        return iter(self.neighbours.items())
    
    def __repr__(self) -> str:
        return f"Node({self.name})"
    
    def __lt__(self, other):
        return self.name < other.name

    
