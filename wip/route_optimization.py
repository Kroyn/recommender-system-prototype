# DESCRIPTION:
# Приклад застосування: Оптимізація маршруту.
import numpy as np
import matplotlib.pyplot as plt

# Цільова функція (знаходимо мінімум)
def sphere_function(x):
    return sum(x**2)

# Параметри алгоритму
POPULATION_SIZE = 50
GENOME_LENGTH = 3
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
GENERATIONS = 100

# 1. Ініціалізація популяції
population = np.random.uniform(-5, 5, (POPULATION_SIZE, GENOME_LENGTH))
best_fitness_history = []
avg_fitness_history = []

print("Запуск еволюційного алгоритму...")
print("Цільова функція: f(x) = sum(x^2)")
print(f"Розмір популяції: {POPULATION_SIZE}, Поколінь: {GENERATIONS}")

for generation in range(GENERATIONS):
    # 2. Оцінка пристосованості
    fitness = np.array([sphere_function(ind) for ind in population])

    # 3. Відбір (турнірний)
    selected_indices = []
    for _ in range(POPULATION_SIZE):
        # Вибираємо 3 випадкових особини і беремо найкращу
        contestants = np.random.choice(POPULATION_SIZE, 3, replace=False)
        winner = contestants[np.argmin(fitness[contestants])]
        selected_indices.append(winner)

    # 4. Кросовер (одноточковий)
    new_population = []
    for i in range(0, POPULATION_SIZE, 2):
        if i+1 < POPULATION_SIZE and np.random.random() < CROSSOVER_RATE:
            parent1 = population[selected_indices[i]]
            parent2 = population[selected_indices[i+1]]
            crossover_point = np.random.randint(1, GENOME_LENGTH)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            new_population.extend([child1, child2])
        else:
            new_population.extend([population[selected_indices[i]], population[selected_indices[i]]])

    # 5. Мутація
    for i in range(POPULATION_SIZE):
        if np.random.random() < MUTATION_RATE:
            mutation_point = np.random.randint(GENOME_LENGTH)
            new_population[i][mutation_point] += np.random.normal(0, 1)

    population = np.array(new_population)

    # Збереження статистики
    best_fitness = np.min(fitness)
    avg_fitness = np.mean(fitness)
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)

    if generation % 20 == 0:
        print(f"Покоління {generation}: Найкраща пристосованість = {best_fitness:.4f}")

# Результати
best_solution = population[np.argmin([sphere_function(ind) for ind in population])]
print(f"\nНайкраще рішення: {best_solution}")
print(f"Значення функції: {sphere_function(best_solution):.6f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history, label='Найкраща пристосованість', linewidth=2)
plt.plot(avg_fitness_history, label='Середня пристосованість', alpha=0.7)
plt.xlabel('Покоління')
plt.ylabel('Пристосованість')
plt.title('Еволюція пристосованості популяції')
plt.legend()
plt.grid(True)
plt.show()
