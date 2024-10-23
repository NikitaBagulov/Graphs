from node import Node
from graph import Graph
import random

import matplotlib.pyplot as plt
from IPython.display import clear_output

class AntAlgorithm:
    def __init__(self, graph: Graph, num_ants: int, num_iterations: int, decay: float, alpha: float, beta: float):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromones = {edge: 1.0 for edge in self.get_all_edges()}
        self.iterations = []
        self.distances = []

    def get_all_edges(self):
        edges = []
        for node in self.graph.nodes:
            for neighbour, weight in node.nb_begin():
                edges.append((node, neighbour))
        return edges

    def run(self):
        best_cycle = None
        best_distance = float('inf')

        plt.ion()
        fig, ax = plt.subplots()

        for iteration in range(self.num_iterations):
            all_cycles = []
            for _ in range(self.num_ants):
                cycle, distance = self.construct_solution()
                if distance < best_distance:
                    best_distance = distance
                    best_cycle = cycle
                    print(f"New best distance found: {best_distance} at iteration {iteration + 1}")
                all_cycles.append((cycle, distance))
            
            self.update_pheromones(all_cycles)

            self.iterations.append(iteration + 1)
            self.distances.append(best_distance)

            clear_output(wait=True)
            ax.clear()
            ax.plot(self.iterations, self.distances, label='Best Distance')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Distance')
            ax.set_title('Ant Algorithm Optimization')
            ax.legend()
            plt.draw()
            plt.pause(0.1)

        plt.ioff()
        plt.show()

        return best_cycle, best_distance

    def construct_solution(self):
        start_node = random.choice(list(self.graph.nodes))
        current_node = start_node
        unvisited_nodes = set(self.graph.nodes)
        unvisited_nodes.remove(current_node)
        
        cycle = [current_node]
        total_distance = 0

        while unvisited_nodes:
            next_node = self.select_next_node(current_node, unvisited_nodes)
            if next_node is None:
                return None, float('inf')

            cycle.append(next_node)
            total_distance += current_node.neighbours[next_node]
            current_node = next_node
            unvisited_nodes.remove(next_node)

        if start_node in current_node.neighbours:
            total_distance += current_node.neighbours[start_node]
            cycle.append(start_node)
        else:
            return None, float('inf')

        return cycle, total_distance

    def select_next_node(self, current_node: Node, unvisited_nodes: set):
        neighbours = [(neighbour, weight) for neighbour, weight in current_node.nb_begin() if neighbour in unvisited_nodes]
        
        if not neighbours:
            return None

        total_pheromone = sum(self.pheromones[(current_node, neighbour)] ** self.alpha * 
                            (1 / weight) ** self.beta for neighbour, weight in neighbours)

        probabilities = []
        for neighbour, weight in neighbours:
            pheromone_level = self.pheromones[(current_node, neighbour)] ** self.alpha
            desirability = (1 / weight) ** self.beta
            probability = pheromone_level * desirability / total_pheromone
            probabilities.append(probability)

        chosen = random.choices(neighbours, weights=probabilities)[0][0]
        return chosen

    def update_pheromones(self, all_cycles):
        for edge in self.pheromones.keys():
            self.pheromones[edge] *= (1 - self.decay)

        for cycle, distance in all_cycles:
            if cycle and len(set(cycle)) == len(self.graph.nodes) + 1:
                pheromone_deposit = 1 / distance
                for i in range(len(cycle) - 1):
                    edge = (cycle[i], cycle[i + 1])
                    if edge in self.pheromones:
                        self.pheromones[edge] += pheromone_deposit


