import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.spatial.distance import cdist
import os

def objective_function(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def compare_gradient_methods():
    x0 = np.array([-1.5, 2.0, 1.5, -1.0, 0.5])
    
    methods = ['CG', 'BFGS', 'L-BFGS-B', 'SLSQP']
    results = {}
    
    print("Порівняння градієнтних методів оптимізації:")
    print("=" * 60)
    
    for method in methods:
        start_time = time.time()
        result = minimize(objective_function, x0, method=method, 
                         options={'disp': False})
        end_time = time.time()
        
        results[method] = {
            'x': result.x,
            'fun': result.fun,
            'nfev': result.nfev,
            'nit': result.nit,
            'time': end_time - start_time,
            'success': result.success
        }
        
        print(f"\n{method}:")
        print(f"  Мінімум: {result.fun:.6f}")
        print(f"  Ітерації: {result.nit}")
        print(f"  Обчислення функції: {result.nfev}")
        print(f"  Час: {end_time - start_time:.4f} сек")
        print(f"  Успіх: {result.success}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    times = [results[m]['time'] for m in methods]
    ax1.bar(methods, times, color=['blue', 'green', 'red', 'orange'])
    ax1.set_title('Час виконання методів')
    ax1.set_ylabel('Час (секунди)')
    
    iterations = [results[m]['nit'] for m in methods]
    ax2.bar(methods, iterations, color=['blue', 'green', 'red', 'orange'])
    ax2.set_title('Кількість ітерацій')
    ax2.set_ylabel('Ітерації')
    
    nfev = [results[m]['nfev'] for m in methods]
    ax3.bar(methods, nfev, color=['blue', 'green', 'red', 'orange'])
    ax3.set_title('Обчислення функції')
    ax3.set_ylabel('Кількість викликів')

    values = [results[m]['fun'] for m in methods]
    ax4.bar(methods, values, color=['blue', 'green', 'red', 'orange'])
    ax4.set_title('Значення функції в мінімумі')
    ax4.set_ylabel('f(x)')
    
    plt.tight_layout()
    plt.show()
    
    return results

class GeneticAlgorithm:
    def __init__(self, population_size=50, generations=100, mutation_rate=0.2):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_history = []
        
    def initialize_population(self, n_features):
        return np.random.uniform(0.01, 5.0, (self.population_size, n_features))
    
    def fitness_function(self, individual, X_train, X_test, y_train, y_test):
        try:
            weights = individual / np.sum(np.abs(individual))
            
            X_train_weighted = X_train * weights
            X_test_weighted = X_test * weights
            
            model = RandomForestRegressor(
                n_estimators=15, 
                max_depth=8,
                random_state=42,
                min_samples_split=5
            )
            model.fit(X_train_weighted, y_train)

            y_pred = model.predict(X_test_weighted)
            mse = mean_squared_error(y_test, y_pred)
            
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            
            fitness = max(0.001, r2 + 1.0)
            diversity = np.std(weights) * 0.1
            fitness += diversity
            
            return fitness
            
        except Exception as e:
            return 0.001
    
    def select_parents(self, population, fitness_scores):
        selected = []
        for _ in range(self.population_size):
            contestants = np.random.choice(len(population), size=5, replace=False)
            best_idx = contestants[np.argmax(fitness_scores[contestants])]
            selected.append(population[best_idx])
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child
    
    def mutate(self, individual):
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                mutation_factor = np.random.uniform(0.5, 2.0)
                individual[i] *= mutation_factor
                individual[i] = np.clip(individual[i], 0.001, 10.0)
        return individual
    
    def optimize_recommendation_weights(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        
        n_features = X.shape[1]
        population = self.initialize_population(n_features)
        
        best_fitness = -np.inf
        best_individual = None
        
        print(f"\nЗапуск генетичного алгоритму...")
        print(f"Розмір популяції: {self.population_size}")
        print(f"Кількість поколінь: {self.generations}")
        print(f"Кількість ознак: {n_features}")
        
        for generation in range(self.generations):
            fitness_scores = []
            for ind in population:
                fitness = self.fitness_function(ind, X_train, X_test, y_train, y_test)
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            current_best_fitness = np.max(fitness_scores)
            current_avg_fitness = np.mean(fitness_scores)
            
            self.fitness_history.append(current_best_fitness)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmax(fitness_scores)].copy()
                improved = True
            else:
                improved = False

            parents = self.select_parents(population, fitness_scores)

            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = parents[i], parents[(i + 1) % len(parents)]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            if best_individual is not None:
                replace_idx = np.random.randint(len(new_population))
                new_population[replace_idx] = best_individual
            
            population = np.array(new_population)[:self.population_size]
            
            if generation % 10 == 0 or improved:
                print(f"Покоління {generation}: Найкраща={current_best_fitness:.4f}, Середня={current_avg_fitness:.4f}")
        
        print(f"Фінальна пристосованість: {best_fitness:.4f}")
        return best_individual, self.fitness_history

def demonstrate_genetic_algorithm():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ratings_path = os.path.join(script_dir, 'ratings.csv')
    movies_path = os.path.join(script_dir, 'movies.csv')
    
    print(f"Шлях до ratings.csv: {ratings_path}")
    print(f"Шлях до movies.csv: {movies_path}")
    
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
  
    merged = ratings.merge(movies, on='movieId')
 
    user_stats = ratings.groupby('userId').agg({
        'rating': ['mean', 'count', 'std']
    }).fillna(0)
    user_stats.columns = ['user_mean_rating', 'user_rating_count', 'user_rating_std']
    
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count', 'std']
    }).fillna(0)
    movie_stats.columns = ['movie_mean_rating', 'movie_rating_count', 'movie_rating_std']

    merged = merged.merge(user_stats, on='userId')
    merged = merged.merge(movie_stats, on='movieId')
    
    feature_columns = ['user_mean_rating', 'user_rating_count', 'user_rating_std',
                      'movie_mean_rating', 'movie_rating_count', 'movie_rating_std']
    
    X = merged[feature_columns].values
    y = merged['rating'].values
    
    print(f"Розмірність даних для GA: {X.shape}")
    print(f"Діапазон цільової змінної: {y.min():.2f} - {y.max():.2f}")

    ga = GeneticAlgorithm(population_size=60, generations=120, mutation_rate=0.25)
    best_weights, fitness_history = ga.optimize_recommendation_weights(X, y)

    normalized_weights = best_weights / np.sum(np.abs(best_weights))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history, 'b-', linewidth=2, marker='o', markersize=3)
    plt.title('Збіжність генетичного алгоритму')
    plt.xlabel('Покоління')
    plt.ylabel('Найкраща пристосованість')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(normalized_weights)), normalized_weights, alpha=0.7, color='green')
    plt.title('Оптимальні ваги ознак (нормалізовані)')
    plt.xlabel('Ознаки')
    plt.ylabel('Нормалізована вага')
    plt.xticks(range(len(feature_columns)), [f'F{i+1}' for i in range(len(feature_columns))], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nРезультати оптимізації:")
    print(f"Найкраща пристосованість: {max(fitness_history):.6f}")
    print(f"Діапазон значень пристосованості: {min(fitness_history):.6f} - {max(fitness_history):.6f}")
    print(f"Оптимальні ваги (нормалізовані):")
    for i, (feature, weight) in enumerate(zip(feature_columns, normalized_weights)):
        print(f"  {feature}: {weight:.4f}")

class AntColonyOptimization:
    def __init__(self, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, 
                 evaporation=0.5, Q=100):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        
    def initialize_pheromones(self, n_items):
        return np.ones((n_items, n_items)) * 0.1
    
    def calculate_similarity(self, items):
        similarity = 1 / (1 + cdist(items.T, items.T, metric='euclidean'))
        np.fill_diagonal(similarity, 0)
        return similarity
    
    def select_next_item(self, current_item, unvisited, pheromone, similarity):
        probabilities = []
        
        for next_item in unvisited:
            pheromone_value = pheromone[current_item, next_item] ** self.alpha
            heuristic_value = similarity[current_item, next_item] ** self.beta
            probabilities.append(pheromone_value * heuristic_value)
        
        probabilities = np.array(probabilities)
        if probabilities.sum() > 0:
            probabilities /= probabilities.sum()
            return np.random.choice(unvisited, p=probabilities)
        else:
            return np.random.choice(unvisited)
    
    def construct_solutions(self, pheromone, similarity, n_items, path_length=10):
        all_paths = []
        all_scores = []
        
        for ant in range(self.n_ants):
            current_item = np.random.randint(n_items)
            path = [current_item]
            unvisited = set(range(n_items)) - {current_item}

            for _ in range(path_length - 1):
                if not unvisited:
                    break
                next_item = self.select_next_item(current_item, list(unvisited), 
                                                pheromone, similarity)
                path.append(next_item)
                unvisited.remove(next_item)
                current_item = next_item

            score = self.evaluate_path(path, similarity)
            all_paths.append(path)
            all_scores.append(score)
        
        return all_paths, all_scores
    
    def evaluate_path(self, path, similarity):
        if len(path) < 2:
            return 0
        total_similarity = 0
        for i in range(len(path) - 1):
            total_similarity += similarity[path[i], path[i+1]]
        return total_similarity / (len(path) - 1)
    
    def update_pheromones(self, pheromone, paths, scores):
        pheromone *= (1 - self.evaporation)

        for path, score in zip(paths, scores):
            for i in range(len(path) - 1):
                pheromone[path[i], path[i+1]] += self.Q * score
                pheromone[path[i+1], path[i]] += self.Q * score
        
        return pheromone
    
    def optimize_recommendation_path(self, items):
        n_items = items.shape[1]
        similarity = self.calculate_similarity(items)
        pheromone = self.initialize_pheromones(n_items)
        
        best_path = None
        best_score = -np.inf
        score_history = []
        
        for iteration in range(self.n_iterations):
            paths, scores = self.construct_solutions(pheromone, similarity, n_items)

            current_best_idx = np.argmax(scores)
            if scores[current_best_idx] > best_score:
                best_score = scores[current_best_idx]
                best_path = paths[current_best_idx]

            pheromone = self.update_pheromones(pheromone, paths, scores)
            
            score_history.append(best_score)
            
            if iteration % 20 == 0:
                print(f"Ітерація {iteration}, Найкращий скор: {best_score:.4f}")
        
        return best_path, best_score, score_history

def demonstrate_ant_colony():
    print("Мурашиний алгоритм: Генерація тестових даних...")
    
    np.random.seed(42)
    n_items = 50

    items_features = np.random.rand(10, n_items)

    aco = AntColonyOptimization(n_ants=15, n_iterations=30)
    best_path, best_score, score_history = aco.optimize_recommendation_path(items_features)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(score_history)
    plt.title('Збіжність мурашиного алгоритму')
    plt.xlabel('Ітерація')
    plt.ylabel('Найкращий скор')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_path, marker='o')
    plt.title('Найкращий шлях рекомендацій')
    plt.xlabel('Позиція в шляху')
    plt.ylabel('ID елементу')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Найкращий шлях (перші 10): {best_path[:10]}")
    print(f"Довжина шляху: {len(best_path)}")
    print(f"Якість шляху: {best_score:.4f}")

if __name__ == "__main__":
    print("ІНТЕЛЕКТУАЛЬНІ МЕТОДИ МОІМС")
    print("=" * 50)
    
    print("\n1. ПОРІВНЯННЯ ГРАДІЄНТНИХ МЕТОДІВ:")
    print("-" * 40)
    try:
        gradient_results = compare_gradient_methods()
    except Exception as e:
        print(f"Помилка в градієнтних методах: {e}")
    
    print("\n2. ГЕНЕТИЧНИЙ АЛГОРИТМ:")
    print("-" * 40)
    try:
        demonstrate_genetic_algorithm()
    except Exception as e:
        print(f"Помилка в генетичному алгоритмі: {e}")

    print("\n3. МУРАШИНИЙ АЛГОРИТМ:")
    print("-" * 40)
    try:
        demonstrate_ant_colony()
    except Exception as e:
        print(f"Помилка в мурашиному алгоритмі: {e}")
    
    print("\n" + "=" * 50)
    print("ВИКОНАННЯ ЗАВЕРШЕНО!")