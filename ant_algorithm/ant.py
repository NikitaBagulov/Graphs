from node import Node
from graph import Graph
import random
import matplotlib.pyplot as plt
import json
import numpy as np

class Ant:
    def __init__(self, graph):
        """
        Инициализация муравья.
        :param graph: Граф, в котором будет работать муравей.
        """
        self.graph = graph
        self.path = []  # Пройденный путь (список узлов)
        self.distance = 0  # Общая длина пройденного пути
        self.probability = 0  # Вероятность выбора пути

    def reset(self):
        """Сбрасывает состояние муравья для новой итерации."""
        self.path = []
        self.distance = 0
        self.probability = 0

class AlphaAnt(Ant):
    def __init__(self, graph):
        super().__init__(graph)
        self.alpha_mode = True
        self.start_node = None

    def select_next_node(self, current_node, unvisited_nodes):
        """
        Выбор следующего узла для альфа-муравья, который не учитывает феромоны, а выбирает кратчайший путь.
        :param current_node: Текущий узел.
        :param unvisited_nodes: Непосещенные узлы.
        :return: Следующий узел.
        """
        # Получаем соседей текущего узла
        neighbours = [(neighbour, weight) for neighbour, weight in current_node.nb_begin() if neighbour in unvisited_nodes]
        
        if not neighbours:
            return None

        next_node = min(neighbours, key=lambda x: x[1])[0]
        return next_node



class AntAlgorithm:
    def __init__(self, graph: Graph, num_ants: int, num_alpha_ants:int, num_iterations: int, decay: float, alpha: float, beta: float):
        """
        Инициализация алгоритма.
        :param graph: Граф (объект класса Graph), содержащий узлы и ребра.
        :param num_ants: Количество муравьев в каждой итерации.
        :param num_alpha_ants: Количество муравьев в каждой итерации.
        :param num_iterations: Максимальное число итераций.
        :param decay: Коэффициент испарения феромонов.
        :param alpha: Вес влияния уровня феромона.
        :param beta: Вес влияния эвристической привлекательности (обратной длине ребра).
        """
        self.graph = graph
        self.num_ants = num_ants
        self.num_alpha_ants = num_alpha_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # Инициализация феромонов для всех ребер графа с минимальным уровнем 0.01
        self.pheromones = {(node, neighbour): 0.01 for node in self.graph.nodes for neighbour, _ in node.nb_begin()}
        
        # Списки для хранения данных о прогрессе
        self.iterations = []  # Номер итерации
        self.distances = []   # Лучшие расстояния на каждой итерации

    def has_hamiltonian_cycle(self):
        """
        Проверка, содержит ли граф хотя бы один гамильтонов цикл с использованием теоремы Ора.
        :return: True, если гамильтонов цикл существует, иначе False.
        """
        n = len(self.graph.nodes)
        if n < 3:
            return False

        nodes_list = list(self.graph.nodes)
        for i, u in enumerate(nodes_list):
            for v in nodes_list[i + 1:]:
                if v not in u.neighbours and u not in v.neighbours:
                    if len(u.neighbours) + len(v.neighbours) >= n:
                        return True
        return False

    def make_hamiltonian(self):
        """
        Добавляет рёбра в граф, чтобы он стал гамильтоновым.
        Сначала соединяет изолированные вершины, затем обновляет список узлов на каждой итерации.
        """
        n = len(self.graph.nodes)
        if n < 3:
            print("Граф слишком мал, чтобы быть гамильтоновым.")
            return []

        nodes_list = list(self.graph.nodes)
        added_edges = []

        # Пока граф не станет гамильтоновым
        while not self.has_hamiltonian_cycle():
            # Находим изолированные вершины
            isolated_nodes = [node for node in nodes_list if len(node.neighbours) == 0]
            non_isolated_nodes = [node for node in nodes_list if len(node.neighbours) > 0]

            # Соединяем изолированные вершины с наиболее связанными
            for isolated in isolated_nodes:
                for non_isolated in sorted(non_isolated_nodes, key=lambda node: len(node.neighbours), reverse=True):
                    if non_isolated not in isolated.neighbours:
                        isolated.add_neighbour(non_isolated, 1.0)
                        self.pheromones[(isolated, non_isolated)] = 0.05
                        added_edges.append((isolated, non_isolated))
                        print(f"Добавлено ребро для изолированной вершины: {isolated} - {non_isolated}")
                        break  # Каждой изолированной вершине добавляем только одно ребро

            # Пересортируем список узлов по количеству соседей
            nodes_list.sort(key=lambda node: len(node.neighbours))

            # Добавляем рёбра для вершин с меньшим числом соседей
            for u in nodes_list:
                for v in nodes_list:
                    if v != u and v not in u.neighbours and u not in v.neighbours:
                        u.add_neighbour(v, 1.0)
                        self.pheromones[(u, v)] = 0.05
                        added_edges.append((u, v))
                        print(f"Добавлено ребро: {u} - {v}")
                        break  # Каждой вершине добавляем только одно ребро за итерацию

            # Проверяем, стал ли граф гамильтоновым
            if self.has_hamiltonian_cycle():
                print("Граф стал гамильтоновым!")
                return added_edges

            print("Граф ещё не стал гамильтоновым. Добавляем новые рёбра.")

        return added_edges

    def make_hamiltonian_simple(self):
        """
        Простейший алгоритм для превращения графа в гамильтонов:
        соединяем любые две несоединённые вершины.
        """
        n = len(self.graph.nodes)
        if n < 3:
            print("Граф слишком мал, чтобы быть гамильтоновым.")
            return []

        added_edges = []

        # Пробегаем по всем парам вершин
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u != v and v not in u.neighbours:
                    # Соединяем несвязанные вершины
                    u.add_neighbour(v, 1.0)
                    self.pheromones[(u, v)] = 0.01  # Обновляем феромоны
                    added_edges.append((u, v))
                    print(f"Добавлено ребро: ({u}, {v}).")

        print("Граф стал гамильтоновым (возможно избыточно).")
        return added_edges

    def get_all_edges(self):
        """
        Получение всех ребер графа.
        :return: Список всех пар (узел, соседний узел).
        """
        edges = []
        for node in self.graph.nodes:
            for neighbour, weight in node.nb_begin():
                edges.append((node, neighbour))
        return edges

    def run(self):
        edges_before = len(self.graph.display_edges())
        if not self.has_hamiltonian_cycle():
            print("Граф не имеет гамильтонова цикла.")
            added_ages = self.make_hamiltonian()
            print(len(added_ages))
            edges_after = len(self.graph.display_edges())
            print(f"Количество рёбер до: {edges_before}")
            print(f"Количество добавленных рёбер: {len(added_ages)}")
            print(f"Количество рёбер после: {edges_after}")
        best_cycle = None  # Хранение лучшего найденного маршрута
        best_distance = float('inf')  # Хранение наименьшей длины маршрута
        no_improvement_count = 0  # Счетчик итераций без улучшений
        max_no_improvement = self.num_iterations  # Порог для завершения алгоритма

        # Настройка интерактивной визуализации
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

        iteration = 0
        probabilities = []  # Массив для хранения вероятностей лучшего пути
        min_paths = []  # Массив для хранения минимальных расстояний
        all_pheromones = []
        all_cycles = []
        best_path_probability_per_iteration = []

        # Создаем обычных и альфа-муравьев
        ants = [Ant(self.graph) for _ in range(self.num_ants)]  # Обычные муравьи
        alpha_ants = [AlphaAnt(self.graph) for _ in range(self.num_alpha_ants)]  # Альфа-муравьи

        while no_improvement_count < max_no_improvement:
            print(iteration)
            iteration += 1
            improved = False

            # Каждый муравей строит решение
            for i, ant in enumerate(ants):
                print(iteration, i)
                cycle, distance = self.construct_solution(ant)
                if distance < best_distance:
                    best_distance = distance
                    best_cycle = cycle
                    print(f"New best distance found: {best_distance} at iteration {iteration}")
                    improved = True

                if cycle is not None:
                    all_cycles.append((cycle, distance))
                    self.update_pheromones(ant)

            # Каждый альфа-муравей строит решение
            for alpha_ant in alpha_ants:
                self.construct_alpha_solution(alpha_ant)
                if alpha_ant.path:  # Если альфа-муравей нашел путь, обновляем феромоны
                    self.update_pheromones(alpha_ant)

            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            self.evaporate_pheromones()
            all_pheromones.append(self.pheromones.copy())
            self.iterations.append(iteration)
            self.distances.append(best_distance)

            # Добавление минимальных путей, вероятностей и других метрик
            if iteration > 1:
                min_distance_in_iteration = min([distance for _, distance in all_cycles]) if all_cycles else min_paths[-1]
            else:
                min_distance_in_iteration = None
            min_paths.append(min_distance_in_iteration)  # Добавляем текущую минимальную длину пути
            probability_of_best_path = self.calculate_probability_of_best_path(best_cycle, self.pheromones)
            probabilities.append(probability_of_best_path)

            if best_cycle is not None:
                product_of_probabilities = 1
                for i in range(len(best_cycle) - 1):
                    current_node = best_cycle[i]
                    next_node = best_cycle[i + 1]
                    probability = self.calculate_transition_probability(current_node, next_node)
                    product_of_probabilities *= probability
                best_path_probability_per_iteration.append(product_of_probabilities)
            else:
                best_path_probability_per_iteration.append(0)

        # Построение графиков
        ax1.clear()
        ax1.plot(self.iterations, self.distances, label='Best Distance')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Distance')
        ax1.set_title('Ant Algorithm Optimization')
        ax1.legend()

        ax2.clear()
        ax2.plot(self.iterations, min_paths, label='Minimum Path Length', color='green')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Minimum Path Length')
        ax2.set_title('Minimum Path Length Over Iterations')
        ax2.legend()

        ax3.clear()
        ax3.plot(self.iterations, probabilities, label='Ratio of Best Path', color='orange')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Ratio of Choosing Best Path')
        ax3.legend()

        # График вероятности лучшего пути
        ax4.clear()
        ax4.plot(self.iterations, best_path_probability_per_iteration, label='Best Path Probability', color='blue')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Probability')
        ax4.set_title('History of Best Path Probability')
        ax4.legend()

        plt.show()

        return best_cycle, best_distance




    def calculate_probability_of_best_path(self, best_cycle, pheromones):
        """
        Расчет вероятности выбора лучшего найденного пути.
        :param best_cycle: Лучший маршрут.
        :return: Вероятность выбора лучшего маршрута.
        """
        if not best_cycle:
            return 0
        
        total_pheromone = sum(
            pheromones[(best_cycle[i], best_cycle[i + 1])] 
            for i in range(len(best_cycle) - 1)
        )
        
        if total_pheromone == 0:
            return 0
        
        probability = total_pheromone / sum(pheromones.values())
        
        return probability

    def construct_solution(self, ant: Ant):
        """
        Построение маршрута для одного муравья.
        :param ant: Экземпляр класса Ant.
        :return: Построенный маршрут (цикл) и его длина.
        """
        start_node = random.choice(list(self.graph.nodes))  # Случайный старт
        current_node = start_node
        unvisited_nodes = set(self.graph.nodes)
        unvisited_nodes.remove(current_node)

        ant.reset()  # Сбрасываем состояние муравья
        ant.path.append(current_node)  # Начало маршрута

        # Пока есть непосещенные узлы
        while unvisited_nodes:
            next_node = self.select_next_node(current_node, unvisited_nodes)
            if next_node is None:
                print("No next node")
                return None, float('inf')  # Невозможно построить маршрут

            # Добавляем узел в маршрут и обновляем длину
            ant.path.append(next_node)
            ant.distance += current_node.neighbours[next_node]
            current_node = next_node
            unvisited_nodes.remove(next_node)

        # Проверка замыкания цикла
        if start_node in current_node.neighbours:

            ant.path.append(start_node)
            ant.distance += current_node.neighbours[start_node]
        else:
            print("No cycle found")
            return None, float('inf')

        return ant.path, ant.distance
    
    def construct_alpha_solution(self, alpha_ant):
        """
        Строит маршрут для альфа-муравья, игнорируя феромоны.
        :param alpha_ant: Альфа-муравей.
        """
        alpha_ant.start_node = random.choice(list(self.graph.nodes))
        alpha_ant.reset()
        current_node = alpha_ant.start_node
        alpha_ant.path.append(current_node)
        unvisited_nodes = set(self.graph.nodes)
        unvisited_nodes.remove(current_node)
        while unvisited_nodes:
            next_node = self.select_next_node(current_node, unvisited_nodes)
            if next_node is None:
                break  # Невозможно построить маршрут

            alpha_ant.path.append(next_node)
            alpha_ant.distance += current_node.neighbours[next_node]
            current_node = next_node
            unvisited_nodes.remove(next_node)

        # Замыкаем цикл, если вернулись к началу
        if alpha_ant.start_node in current_node.neighbours:
            alpha_ant.path.append(alpha_ant.start_node)
            alpha_ant.distance += current_node.neighbours[alpha_ant.start_node]
        else:
            return None, float('inf')


    def select_next_node(self, current_node: Node, unvisited_nodes: set):
        """
        Выбор следующего узла на основе вероятностей.
        :param current_node: Текущий узел.
        :param unvisited_nodes: Непосещенные узлы.
        :return: Следующий узел.
        """
        # Получение соседей текущего узла, которые ещё не посещены
        neighbours = [(neighbour, weight) for neighbour, weight in current_node.nb_begin() if neighbour in unvisited_nodes]
        
        if not neighbours:
            print("No neighbours")
            return None  # Если нет доступных соседей, маршрут завершить нельзя

        # Расчет общей суммы "привлекательностей" переходов
        total_pheromone = sum(
            self.pheromones[(current_node, neighbour)] ** self.alpha * (1 / weight) ** self.beta 
            for neighbour, weight in neighbours
        )
        # Если феромоны отсутствуют, выбираем случайного соседа
        if total_pheromone == 0:
            return random.choice(neighbours)[0]

        # Вычисление вероятностей для каждого соседа
        probabilities = []
        for neighbour, weight in neighbours:
            pheromone_level = self.pheromones[(current_node, neighbour)] ** self.alpha
            desirability = (1 / weight) ** self.beta
            probability = pheromone_level * desirability / total_pheromone
            probabilities.append(probability)
        # Случайный выбор соседа с учетом вероятностей
        chosen = random.choices(neighbours, weights=probabilities)[0][0] #[(node_A, 10), (node_B, 20), (node_C, 15)] [0.2, 0.5, 0.3]
        return chosen
    def calculate_transition_probability(self, current_node, next_node):
        """
        Расчет вероятности перехода между двумя узлами.
        :param current_node: Текущий узел.
        :param next_node: Следующий узел.
        :return: Вероятность перехода.
        """
        pheromone_level = self.pheromones[(current_node, next_node)] ** self.alpha
        weight = current_node.neighbours[next_node]
        desirability = (1 / weight) ** self.beta
        total_pheromone = sum(
            self.pheromones[(current_node, neighbour)] ** self.alpha * (1 / current_node.neighbours[neighbour]) ** self.beta
            for neighbour in current_node.neighbours
        )
        probability = pheromone_level * desirability / total_pheromone
        return probability

    def evaporate_pheromones(self):
        """
        Испарение феромонов.
        """
        # Испарение феромонов
        for edge in self.pheromones.keys():
            self.pheromones[edge] *= (1 - self.decay)


    def update_pheromones(self, ant: Ant):
        """
        Обновление уровня феромонов на всех ребрах пути муравья.
        :param ant: Экземпляр класса Ant.
        """
        if ant.path and len(ant.path) == len(self.graph.nodes) + 1:  # Проверка на замкнутый цикл
            pheromone_deposit = 1 / ant.distance if ant.distance != 0 else 0
            unique_edges = set()

            # Обновление феромонов только для уникальных ребер
            for i in range(len(ant.path) - 1):
                edge = (ant.path[i], ant.path[i + 1])
                if edge not in unique_edges and edge in self.pheromones:
                    self.pheromones[edge] += pheromone_deposit
                    unique_edges.add(edge)

    @classmethod
    def from_config(cls, config_file: str, graph: Graph):
        """
        Создание экземпляра алгоритма из конфигурационного файла.
        :param config_file: JSON-файл с настройками.
        :param graph: Граф для оптимизации.
        :return: Экземпляр AntAlgorithm.
        """
        with open(config_file, 'r') as file:
            config = json.load(file)
        
        return cls(
            graph=graph,
            num_ants=config.get("num_ants", 1),
            num_alpha_ants=config.get("num_alpha_ants",1),
            num_iterations=config.get("num_iterations", 100),
            decay=config.get("decay", 0.1),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 2.0)
        )
