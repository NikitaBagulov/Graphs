from node import Node
from graph import Graph
import random
import matplotlib.pyplot as plt
import json

class AntAlgorithm:
    def __init__(self, graph: Graph, num_ants: int, num_iterations: int, decay: float, alpha: float, beta: float):
        """
        Инициализация алгоритма.
        :param graph: Граф (объект класса Graph), содержащий узлы и ребра.
        :param num_ants: Количество муравьев в каждой итерации.
        :param num_iterations: Максимальное число итераций.
        :param decay: Коэффициент испарения феромонов.
        :param alpha: Вес влияния уровня феромона.
        :param beta: Вес влияния эвристической привлекательности (обратной длине ребра).
        """
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # Инициализация феромонов для всех ребер графа с минимальным уровнем 0.01
        self.pheromones = {(node, neighbour): 0.01 for node in self.graph.nodes for neighbour, _ in node.nb_begin()}
        
        # Списки для хранения данных о прогрессе
        self.iterations = []  # Номер итерации
        self.distances = []   # Лучшие расстояния на каждой итерации

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
        """
        Основной метод для запуска алгоритма.
        :return: Лучший маршрут (цикл) и его длина.
        """
        best_cycle = None  # Хранение лучшего найденного маршрута
        best_distance = float('inf')  # Хранение наименьшей длины маршрута
        no_improvement_count = 0  # Счетчик итераций без улучшений
        max_no_improvement = self.num_iterations  # Порог для завершения алгоритма

        # Настройка интерактивной визуализации
        # plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))

        iteration = 0
        probabilities = []  # Массив для хранения вероятностей
        min_paths = []  # Массив для хранения минимальных 
        all_pheromones = []

        while no_improvement_count < max_no_improvement:
            iteration += 1
            all_cycles = []  # Список маршрутов, построенных всеми муравьями
            improved = False

            # Цикл для муравьев
            for _ in range(self.num_ants):
                cycle, distance = self.construct_solution()
                
                # Обновляем данные
                if distance < best_distance:
                    best_distance = distance
                    best_cycle = cycle
                    print(f"New best distance found: {best_distance} at iteration {iteration}")
                    improved = True
                
                if cycle is not None:
                    all_cycles.append((cycle, distance))
                    self.update_pheromones(cycle, distance)

            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # Сброс счетчика при улучшении
            # Испарение феромонов и запись данных для графиков
            self.evaporate_pheromones()
            all_pheromones.append(self.pheromones.copy())
            self.iterations.append(iteration)
            self.distances.append(best_distance)
            if iteration>1:
                min_distance_in_iteration = min([distance for _, distance in all_cycles]) if all_cycles else min_paths[-1]
            else:
                min_distance_in_iteration = None
            min_paths.append(min_distance_in_iteration)  # Добавляем текущую минимальную длину пути
            # Обновление вероятности выбора лучшего маршрута
            probability_of_best_path = self.calculate_probability_of_best_path(best_cycle, self.pheromones)
            probabilities.append(probability_of_best_path)

            # Обновление графиков визуализации
            # ax1.clear()
            # ax1.plot(self.iterations, self.distances, label='Best Distance')
            # ax1.set_xlabel('Iteration')
            # ax1.set_ylabel('Distance')
            # ax1.set_title('Ant Algorithm Optimization')
            # ax1.legend()

            

            # ax2.clear()
            # ax2.plot(self.iterations, min_paths, label='Minimum Path Length', color='green')
            # ax2.set_xlabel('Iteration')
            # ax2.set_ylabel('Minimum Path Length')
            # ax2.set_title('Minimum Path Length Over Iterations')
            # ax2.legend()

            # plt.draw()
            # plt.pause(0.1)
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
        best_path_probabilities = [self.calculate_probability_of_best_path(best_cycle, pheromones) for pheromones in all_pheromones]
        ax3.clear()
        ax3.plot(self.iterations, best_path_probabilities, label='Ratio of Best Path', color='orange')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Ratio of Choosing Best Path')
        ax3.legend()
        # plt.ioff()
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

    def construct_solution(self):
        """
        Построение маршрута для одного муравья.
        :return: Построенный маршрут (цикл) и его длина.
        """
        # Выбор случайного начального узла
        start_node = random.choice(list(self.graph.nodes))
        current_node = start_node
        unvisited_nodes = set(self.graph.nodes)
        unvisited_nodes.remove(current_node)
        
        cycle = [current_node]  # Начало маршрута
        total_distance = 0  # Суммарная длина маршрута

        # Пока есть непосещенные узлы
        while unvisited_nodes:
            next_node = self.select_next_node(current_node, unvisited_nodes)
            
            if next_node is None:  # Если невозможно продолжить маршрут
                return None, float('inf')

            # Добавление узла в маршрут и обновление длины
            cycle.append(next_node)
            total_distance += current_node.neighbours[next_node]
            current_node = next_node
            unvisited_nodes.remove(next_node)

        # Проверка возможности замкнуть цикл
        if start_node in current_node.neighbours:
            total_distance += current_node.neighbours[start_node]
            cycle.append(start_node)
        else:
            return None, float('inf')

        return cycle, total_distance

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

    def evaporate_pheromones(self):
        """
        Испарение феромонов.
        """
        # Испарение феромонов
        for edge in self.pheromones.keys():
            self.pheromones[edge] *= (1 - self.decay)


    def update_pheromones(self, cycle, distance):
        """
        Обновление уровня феромонов на всех ребрах пути.
        :param cycle: Маршрут, построенный муравьем.
        :param distance: Длина маршрута, построенного муравьем.
        """
        # Добавление феромонов для маршрутов
        # for cycle, distance in all_cycles:
        if cycle and len(cycle) == len(self.graph.nodes) + 1:  # Проверка на замкнутый цикл
            pheromone_deposit = 1 / distance if distance != 0 else 0  # Количество феромона зависит от длины маршрута
            unique_edges = set()

            # Обновление феромонов только для уникальных ребер
            for i in range(len(cycle) - 1):
                edge = (cycle[i], cycle[i + 1])
                
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
            num_iterations=config.get("num_iterations", 100),
            decay=config.get("decay", 0.1),
            alpha=config.get("alpha", 1.0),
            beta=config.get("beta", 2.0)
        )
