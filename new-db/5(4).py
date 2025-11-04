import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import gzip
import random
import os
from datetime import datetime

class FriendsterMovieAnalyzer:
    def __init__(self):
        self.G = nx.Graph()
        
        print("ЗАВАНТАЖЕННЯ ДАНИХ")
        
        try:
            self.movies_df = pd.read_csv('movies.csv', encoding='utf-8', quotechar='"', on_bad_lines='skip')
            print(f"Завантажено фільмів: {len(self.movies_df)}")
        except:
            self.movies_df = pd.DataFrame({
                'movieId': range(1, 101),
                'title': [f'Movie {i}' for i in range(1, 101)],
                'genres': ['Action|Drama' for _ in range(100)]
            })
            print(f"Створено тестових фільмів: {len(self.movies_df)}")
        
        try:
            self.ratings_df = pd.read_csv('ratings.csv', encoding='utf-8')
            self.ratings_df['date'] = pd.to_datetime(self.ratings_df['date'])
            print(f"Завантажено рейтингів: {len(self.ratings_df)}")
        except:
            self.ratings_df = pd.DataFrame({
                'userId': np.random.randint(1, 101, 1000),
                'movieId': np.random.randint(1, 51, 1000),
                'rating': np.random.uniform(3.0, 5.0, 1000).round(1),
                'timestamp': np.random.randint(964982703, 964984050, 1000),
                'date': pd.date_range('2020-01-01', periods=1000, freq='D')
            })
            print(f"Створено тестових рейтингів: {len(self.ratings_df)}")
        
        try:
            self.users_df = pd.read_csv('users.csv', encoding='utf-8')
            print(f"Завантажено користувачів: {len(self.users_df)}")
        except:
            self.users_df = pd.DataFrame({
                'userId': range(1, 101),
                'age': np.random.randint(18, 60, 100),
                'gender': np.random.choice(['M', 'F'], 100),
                'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor'], 100)
            })
            print(f"Створено тестових користувачів: {len(self.users_df)}")
        
        os.makedirs('media', exist_ok=True)
        
    def generate_synchronized_dataset(self):
        user_ids = list(self.users_df['userId'].unique())
        edges = []
        
        print(f"Генерація мережі для {len(user_ids)} користувачів...")
        
        for user_id in user_ids:
            num_friends = random.randint(3, 15)
            friends = random.sample([uid for uid in user_ids if uid != user_id], num_friends)
            for friend in friends:
                edges.append((user_id, friend))
        
        with gzip.open('com-friendster.ungraph.txt.gz', 'wt') as f:
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")
        
        print(f"Згенеровано {len(edges)} ребер мережі")
        return edges
    
    def analyze_network(self):
        edges = self.generate_synchronized_dataset()
        self.G.add_edges_from(edges)
        
        print("\nАНАЛІЗ МЕРЕЖІ FRIENDSTER")
        print(f"Кількість вузлів: {self.G.number_of_nodes()}")
        print(f"Кількість ребер: {self.G.number_of_edges()}")
        
        degrees = [deg for _, deg in self.G.degree()]
        print(f"Середній ступінь: {np.mean(degrees):.2f}")
        print(f"Максимальний ступінь: {max(degrees)}")
        print(f"Мінімальний ступінь: {min(degrees)}")
        
        print(f"Коефіцієнт кластеризації: {nx.average_clustering(self.G):.4f}")
        print(f"Щільність мережі: {nx.density(self.G):.6f}")
        
        if nx.is_connected(self.G):
            print(f"Діаметр мережі: {nx.diameter(self.G)}")
        else:
            components = list(nx.connected_components(self.G))
            largest_cc = max(components, key=len)
            subgraph = self.G.subgraph(largest_cc)
            print(f"Кількість компонентів зв'язності: {len(components)}")
            print(f"Розмір найбільшого компоненту: {len(largest_cc)}")
            print(f"Діаметр найбільшого компоненту: {nx.diameter(subgraph)}")
    
    def calculate_centrality(self):
        print("\nРОЗРАХУНОК ЦЕНТРАЛЬНОСТІ")
        
        degree_centrality = nx.degree_centrality(self.G)
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        largest_cc = max(nx.connected_components(self.G), key=len)
        subgraph = self.G.subgraph(largest_cc)
        closeness_centrality = nx.closeness_centrality(subgraph)
        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        betweenness_centrality = nx.betweenness_centrality(self.G, k=min(100, self.G.number_of_nodes()))
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("ТОП-10 КОРИСТУВАЧІВ ЗА ЦЕНТРАЛЬНІСТЮ:")
        print("Ступінь центральності:", [node for node, _ in top_degree])
        print("Близькість центральності:", [node for node, _ in top_closeness])
        print("Посередництво центральності:", [node for node, _ in top_betweenness])
        
        return degree_centrality, closeness_centrality, betweenness_centrality
    
    def find_shortest_path(self, source, target):
        try:
            path = nx.shortest_path(self.G, source=source, target=target)
            length = nx.shortest_path_length(self.G, source=source, target=target)
            print(f"\nНайкоротший шлях між {source} та {target}: {path}")
            print(f"Довжина шляху: {length}")
            return path
        except nx.NetworkXNoPath:
            print(f"Шлях між {source} та {target} не існує")
            return None
        except nx.NodeNotFound:
            print(f"Один з користувачів ({source} або {target}) не знайдений у мережі")
            return None
    
    def friend_recommendations(self, user_id, top_k=10):
        if user_id not in self.G:
            print(f"Користувач {user_id} не знайдений у мережі")
            return []
        
        neighbors = set(self.G.neighbors(user_id))
        recommendations = {}
        
        for neighbor in neighbors:
            neighbor_neighbors = set(self.G.neighbors(neighbor))
            common = neighbors.intersection(neighbor_neighbors)
            
            for common_neighbor in common:
                if common_neighbor != user_id and common_neighbor not in neighbors:
                    jaccard = len(common) / len(neighbors.union(neighbor_neighbors)) if neighbors.union(neighbor_neighbors) else 0
                    if common_neighbor not in recommendations:
                        recommendations[common_neighbor] = 0
                    recommendations[common_neighbor] += jaccard
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        print(f"\nРекомендації друзів для користувача {user_id}:")
        if sorted_recommendations:
            for user, score in sorted_recommendations:
                print(f"Користувач {user}: {score:.4f}")
        else:
            print("Немає рекомендацій на основі спільних сусідів")
        
        return [user for user, _ in sorted_recommendations]
    
    def movie_recommendations_based_on_friends(self):
        user_movie_ratings = self.ratings_df.groupby('userId')['movieId'].apply(list).to_dict()
        
        recommendations = {}
        analyzed_users = 0
        
        for user_id in list(self.G.nodes()):
            if user_id in user_movie_ratings and analyzed_users < 50:
                user_movies = set(user_movie_ratings[user_id])
                friend_recommendations = []
                
                for friend in self.G.neighbors(user_id):
                    if friend in user_movie_ratings:
                        friend_movies = set(user_movie_ratings[friend])
                        new_movies = friend_movies - user_movies
                        friend_recommendations.extend(new_movies)
                
                if friend_recommendations:
                    movie_counts = Counter(friend_recommendations)
                    top_movies = [movie for movie, _ in movie_counts.most_common(5)]
                    recommendations[user_id] = top_movies
                
                analyzed_users += 1
        
        print("\nРЕКОМЕНДАЦІЇ ФІЛЬМІВ НА ОСНОВІ ДРУЗІВ (ПЕРШІ 10)")
        print()
        
        for user_id, movies in list(recommendations.items())[:10]:
            movie_titles = []
            for movie_id in movies:
                movie_title = self.movies_df[self.movies_df['movieId'] == movie_id]['title']
                if not movie_title.empty:
                    movie_titles.append(movie_title.iloc[0])
                else:
                    movie_titles.append(f"Фільм {movie_id}")
            
            print(f"Користувач {user_id}:")
            print(f"  Рекомендовані фільми: {movie_titles}")
            print()
        
        return recommendations
    
    def visualize_network(self, num_nodes=50):
        if num_nodes > self.G.number_of_nodes():
            num_nodes = self.G.number_of_nodes()
        
        nodes = list(self.G.nodes())[:num_nodes]
        subgraph = self.G.subgraph(nodes)
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        degree_centrality = nx.degree_centrality(subgraph)
        node_sizes = [500 + 2000 * degree_centrality[node] for node in nodes]
        node_colors = [degree_centrality[node] for node in nodes]
        
        nodes_draw = nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                                          node_color=node_colors, cmap='plasma', 
                                          alpha=0.8, edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, edge_color='gray', width=0.5)
        
        important_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = {node: str(node) for node, _ in important_nodes}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
        
        plt.colorbar(nodes_draw, label='Ступінь центральності')
        plt.title(f'Мережа Friendster ({num_nodes} користувачів)', fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('media/friendster_network.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_communities(self):
        from networkx.algorithms import community
        
        nodes_sample = list(self.G.nodes())[:min(500, self.G.number_of_nodes())]
        subgraph = self.G.subgraph(nodes_sample)
        
        try:
            communities = list(community.greedy_modularity_communities(subgraph, resolution=1.2))
            
            print(f"\nСТРУКТУРА СПІЛЬНОТ")
            print(f"Кількість спільнот: {len(communities)}")
            
            community_sizes = [len(comm) for comm in communities]
            print(f"Розміри спільнот: {community_sizes}")
            print(f"Середній розмір спільноти: {np.mean(community_sizes):.1f}")
            print(f"Модулярність: {community.modularity(subgraph, communities):.4f}")
            
            return communities
        except Exception as e:
            print(f"Помилка при аналізі спільнот: {e}")
            return []
    
    def create_recommendation_analysis(self):
        users_sample = list(self.G.nodes())[:15]
        
        metrics_data = []
        for user in users_sample:
            if user in self.G:
                degree = self.G.degree(user)
                clustering = nx.clustering(self.G, user)
                metrics_data.append({
                    'user': user,
                    'degree': degree,
                    'clustering': clustering
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(df_metrics['user'].astype(str), df_metrics['degree'], color='skyblue', alpha=0.7)
        plt.title('Ступінь користувачів', fontsize=12)
        plt.xlabel('Користувач')
        plt.ylabel('Ступінь')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(df_metrics['user'].astype(str), df_metrics['clustering'], color='lightcoral', alpha=0.7)
        plt.title('Коефіцієнт кластеризації', fontsize=12)
        plt.xlabel('Користувач')
        plt.ylabel('Кластеризація')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('media/recommendation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_final_report(self):
        report = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'density': nx.density(self.G),
            'clustering': nx.average_clustering(self.G),
            'assortativity': nx.degree_assortativity_coefficient(self.G) if self.G.number_of_edges() > 0 else 0
        }
        
        print("\nДЕТАЛЬНИЙ ЗВІТ ПРО МЕРЕЖУ")
        for key, value in report.items():
            print(f"{key}: {value:.4f}")
        
        degrees = [deg for _, deg in self.G.degree()]
        print(f"\nДодаткові статистики:")
        print(f"Медіана ступеня: {np.median(degrees):.2f}")
        print(f"Стандартне відхилення ступеня: {np.std(degrees):.2f}")
        
        return report

def main():
    print("Запуск аналізу мережі Friendster...")
    
    analyzer = FriendsterMovieAnalyzer()
    
    analyzer.analyze_network()
    
    analyzer.calculate_centrality()
    
    analyzer.find_shortest_path(1, 50)
    analyzer.find_shortest_path(2, 25)
    
    analyzer.friend_recommendations(1)
    analyzer.friend_recommendations(10)
    
    analyzer.movie_recommendations_based_on_friends()
    
    analyzer.visualize_network(50)
    
    analyzer.analyze_communities()
    
    analyzer.create_recommendation_analysis()
    
    analyzer.generate_final_report()
    
    print("\nАНАЛІЗ УСПІШНО ЗАВЕРШЕНО")

if __name__ == "__main__":
    main()