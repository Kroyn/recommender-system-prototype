# DESCRIPTION:
# Порівняння методу мурашиних колоній з іншими методами
import numpy as np
import matplotlib.pyplot as plt


class AntColonyOptimization:
    def __init__(self, distances, n_ants=20, n_iterations=100,
                 evaporation_rate=0.5, alpha=1, beta=2):
        self.distances = distances
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # Вага феромону
        self.beta = beta    # Вага відстані
        self.n_cities = len(distances)
        self.pheromone = np.ones((self.n_cities, self.n_cities))

    def run(self):
        best_path = None
        best_distance = float('inf')
        history = []

        for iteration in range(self.n_iterations):
            paths = self._construct_solutions()
            distances = [self._calculate_distance(path) for path in paths]

            # Оновлення найкращого рішення
            min_distance = min(distances)
            if min_distance < best_distance:
                best_distance = min_distance
                best_path = paths[np.argmin(distances)]

            self._update_pheromone(paths, distances)
            history.append(best_distance)

            if iteration % 20 == 0:
                print(f"Ітерація {iteration}: Найкраща відстань = {best_distance:.2f}")

        return best_path, best_distance, history

    def _construct_solutions(self):
        paths = []
        for _ in range(self.n_ants):
            path = [0]  # Починаємо з міста 0
            unvisited = list(range(1, self.n_cities))

            while unvisited:
                current_city = path[-1]
                probabilities = self._calculate_probabilities(current_city, unvisited)
                next_city = np.random.choice(unvisited, p=probabilities)
                path.append(next_city)
                unvisited.remove(next_city)

            paths.append(path)
        return paths

    def _calculate_probabilities(self, current_city, unvisited):
        probabilities = []
        for city in unvisited:
            pheromone = self.pheromone[current_city][city] ** self.alpha
            visibility = (1.0 / self.distances[current_city][city]) ** self.beta
            probabilities.append(pheromone * visibility)

        probabilities = np.array(probabilities)
        return probabilities / probabilities.sum()

    def _calculate_distance(self, path):
        total = 0
        for i in range(len(path)):
            total += self.distances[path[i-1]][path[i]]
        return total

    def _update_pheromone(self, paths, distances):
        # Випаровування
        self.pheromone *= (1 - self.evaporation_rate)

        # Додавання нового феромону
        for path, distance in zip(paths, distances):
            pheromone_to_add = 1.0 / distance
            for i in range(len(path)):
                self.pheromone[path[i-1]][path[i]] += pheromone_to_add
                self.pheromone[path[i]][path[i-1]] += pheromone_to_add


# Створюємо тестові дані (відстані між 10 містами)
np.random.seed(42)
n_cities = 10
cities = np.random.rand(n_cities, 2) * 100


# Обчислюємо матрицю відстаней
distances = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        distances[i][j] = np.linalg.norm(cities[i] - cities[j])


print("Запуск методу мурашиних колоній для задачі комівояжера...")
print(f"Кількість міст: {n_cities}")
print(f"Кількість мурах: {20}, Ітерацій: {100}")


# Запускаємо алгоритм
aco = AntColonyOptimization(distances, n_ants=20, n_iterations=100)
best_path, best_distance, history = aco.run()


print(f"\nНайкращий знайдений маршрут: {best_path}")
print(f"Загальна відстань: {best_distance:.2f}")


# Візуалізація результатів
plt.figure(figsize=(15, 5))


# 1. Візуалізація міст та маршруту
plt.subplot(1, 2, 1)
plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100)
for i, (x, y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12, ha='center', va='center')


# Малюємо маршрут
route = cities[best_path]
route = np.vstack([route, route[0]])  # Замикаємо маршрут
plt.plot(route[:, 0], route[:, 1], 'b-', alpha=0.7)
plt.title(f'Найкращий маршрут (відстань: {best_distance:.2f})')
plt.xlabel('X координата')
plt.ylabel('Y координата')


# 2. Графік збіжності
plt.subplot(1, 2, 2)
plt.plot(history, 'g-', linewidth=2)
plt.xlabel('Ітерація')
plt.ylabel('Найкраща відстань')
plt.title('Збіжність алгоритму')
plt.grid(True)
plt.tight_layout()
plt.show()
