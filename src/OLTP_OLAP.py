import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime, timedelta

class MovieRecommendationDatabase:
    def __init__(self):
        self.db_path = 'movie_recommendation_system.db'
        os.makedirs('media', exist_ok=True)
        
    def load_existing_data(self):
        print("Завантаження існуючих даних...")
        
        self.users_df = pd.read_csv('users.csv')
        self.movies_df = pd.read_csv('movies.csv', encoding='utf-8', on_bad_lines='skip', quotechar='"')
        self.ratings_df = pd.read_csv('ratings.csv')
        
        print(f"Користувачі: {len(self.users_df)} записів")
        print(f"Фільми: {len(self.movies_df)} записів")
        print(f"Рейтинги: {len(self.ratings_df)} записів")
        
    def create_database_schema(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DROP TABLE IF EXISTS users')
        cursor.execute('DROP TABLE IF EXISTS movies')
        cursor.execute('DROP TABLE IF EXISTS ratings')
        cursor.execute('DROP TABLE IF EXISTS friendster_network')
        cursor.execute('DROP TABLE IF EXISTS user_behavior')
        
        cursor.execute('''
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                occupation TEXT NOT NULL,
                registration_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE movies (
                movie_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                genres TEXT NOT NULL,
                release_year INTEGER,
                clean_title TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE ratings (
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                date TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, movie_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE friendster_network (
                user_id1 INTEGER,
                user_id2 INTEGER,
                connection_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id1, user_id2),
                FOREIGN KEY (user_id1) REFERENCES users(user_id),
                FOREIGN KEY (user_id2) REFERENCES users(user_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE user_behavior (
                behavior_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                action_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration_seconds INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Схему бази даних створено")
        
    def populate_database(self):
        conn = sqlite3.connect(self.db_path)
        
        self.movies_df['release_year'] = self.movies_df['title'].str.extract(r'\((\d{4})\)').fillna(2000).astype(int)
        self.movies_df['clean_title'] = self.movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        
        self.users_df.rename(columns={'userId': 'user_id'}).to_sql('users', conn, if_exists='replace', index=False)
        
        movies_clean = self.movies_df[['movieId', 'clean_title', 'genres', 'release_year']].rename(
            columns={'movieId': 'movie_id', 'clean_title': 'title'})
        movies_clean.to_sql('movies', conn, if_exists='replace', index=False)
        
        ratings_clean = self.ratings_df.rename(columns={
            'userId': 'user_id', 
            'movieId': 'movie_id'
        })
        ratings_clean.to_sql('ratings', conn, if_exists='replace', index=False)
        
        user_ids = self.users_df['userId'].tolist()
        friendster_data = []
        for user_id in user_ids[:100]:
            num_friends = np.random.randint(2, 8)
            friends = np.random.choice(
                [uid for uid in user_ids if uid != user_id], 
                num_friends, 
                replace=False
            )
            for friend in friends:
                friendster_data.append({'user_id1': user_id, 'user_id2': friend})
        
        friendster_df = pd.DataFrame(friendster_data)
        friendster_df.to_sql('friendster_network', conn, if_exists='replace', index=False)
        
        behavior_data = []
        for _, rating in self.ratings_df.iterrows():
            user_id = rating['userId']
            movie_id = rating['movieId']
            
            behavior_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'action_type': 'view',
                'duration_seconds': np.random.randint(300, 7200)
            })
            
            behavior_data.append({
                'user_id': user_id,
                'movie_id': movie_id,
                'action_type': 'rate'
            })
        
        behavior_df = pd.DataFrame(behavior_data)
        behavior_df.to_sql('user_behavior', conn, if_exists='replace', index=False)
        
        conn.close()
        print("Базу даних заповнено")

        
        
    def analyze_database(self):
        conn = sqlite3.connect(self.db_path)
        
        print("\n=== АНАЛІЗ БАЗИ ДАНИХ ===")
        
        tables = ['users', 'movies', 'ratings', 'friendster_network', 'user_behavior']
        for table in tables:
            count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn)['count'].iloc[0]
            print(f"{table}: {count} записів")
        
        rating_stats = pd.read_sql('''
            SELECT 
                rating,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ratings), 2) as percentage
            FROM ratings 
            GROUP BY rating 
            ORDER BY rating
        ''', conn)
        print(f"\nРозподіл рейтингів:\n{rating_stats}")
        
        genre_stats = pd.read_sql('''
            SELECT 
                m.genres,
                COUNT(DISTINCT m.movie_id) as movie_count,
                COUNT(*) as rating_count,
                ROUND(AVG(r.rating), 2) as avg_rating
            FROM movies m
            LEFT JOIN ratings r ON m.movie_id = r.movie_id
            GROUP BY m.genres
            ORDER BY rating_count DESC
            LIMIT 10
        ''', conn)
        print(f"\nТоп-10 жанрів:\n{genre_stats}")
        
        user_stats = pd.read_sql('''
            SELECT 
                gender,
                COUNT(*) as user_count,
                ROUND(AVG(age), 1) as avg_age,
                COUNT(DISTINCT occupation) as unique_occupations
            FROM users
            GROUP BY gender
        ''', conn)
        print(f"\nСтатистика користувачів:\n{user_stats}")
        
        network_stats = pd.read_sql('''
            SELECT 
                COUNT(*) as total_connections,
                COUNT(DISTINCT user_id1) as users_with_connections,
                ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT user_id1), 2) as avg_connections_per_user
            FROM friendster_network
        ''', conn)
        print(f"\nСтатистика мережі:\n{network_stats}")
        
        conn.close()
        
    def create_visualizations(self):
        conn = sqlite3.connect(self.db_path)
        
        plt.figure(figsize=(16, 12))
        
        rating_dist = pd.read_sql('''
            SELECT rating, COUNT(*) as count 
            FROM ratings 
            GROUP BY rating 
            ORDER BY rating
        ''', conn)
        plt.subplot(2, 3, 1)
        plt.bar(rating_dist['rating'], rating_dist['count'], color='lightblue', alpha=0.7)
        plt.title('Розподіл рейтингів')
        plt.xlabel('Рейтинг')
        plt.ylabel('Кількість')
        plt.grid(True, alpha=0.3)
        
        genre_popularity = pd.read_sql('''
            SELECT 
                SUBSTR(genres, 1, 20) as genre_group,
                COUNT(*) as rating_count
            FROM ratings r
            JOIN movies m ON r.movie_id = m.movie_id
            GROUP BY genre_group
            ORDER BY rating_count DESC
            LIMIT 8
        ''', conn)
        plt.subplot(2, 3, 2)
        plt.bar(genre_popularity['genre_group'], genre_popularity['rating_count'], color='lightcoral', alpha=0.7)
        plt.title('Популярність жанрів')
        plt.xlabel('Жанри')
        plt.ylabel('Кількість оцінок')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        age_distribution = pd.read_sql('''
            SELECT age, COUNT(*) as user_count
            FROM users
            GROUP BY age
            ORDER BY age
        ''', conn)
        plt.subplot(2, 3, 3)
        plt.plot(age_distribution['age'], age_distribution['user_count'], marker='o', color='green')
        plt.title('Розподіл користувачів за віком')
        plt.xlabel('Вік')
        plt.ylabel('Кількість користувачів')
        plt.grid(True, alpha=0.3)
        
        user_activity = pd.read_sql('''
            SELECT user_id, COUNT(*) as rating_count
            FROM ratings
            GROUP BY user_id
            ORDER BY rating_count DESC
            LIMIT 15
        ''', conn)
        plt.subplot(2, 3, 4)
        plt.bar(user_activity['user_id'].astype(str), user_activity['rating_count'], color='orange', alpha=0.7)
        plt.title('Активність користувачів (топ-15)')
        plt.xlabel('ID користувача')
        plt.ylabel('Кількість оцінок')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        connection_dist = pd.read_sql('''
            SELECT user_id1, COUNT(*) as friend_count
            FROM friendster_network
            GROUP BY user_id1
            ORDER BY friend_count DESC
            LIMIT 10
        ''', conn)
        plt.subplot(2, 3, 5)
        plt.bar(connection_dist['user_id1'].astype(str), connection_dist['friend_count'], color='purple', alpha=0.7)
        plt.title('Кількість друзів (топ-10)')
        plt.xlabel('ID користувача')
        plt.ylabel('Кількість друзів')
        plt.grid(True, alpha=0.3)
        
        year_distribution = pd.read_sql('''
            SELECT release_year, COUNT(*) as movie_count
            FROM movies
            WHERE release_year IS NOT NULL
            GROUP BY release_year
            ORDER BY release_year
        ''', conn)
        plt.subplot(2, 3, 6)
        plt.plot(year_distribution['release_year'], year_distribution['movie_count'], marker='s', color='red')
        plt.title('Розподіл фільмів за роками')
        plt.xlabel('Рік випуску')
        plt.ylabel('Кількість фільмів')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('media/database_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.create_3d_visualizations(conn)
        conn.close()
        
    def create_3d_visualizations(self, conn):
        fig = plt.figure(figsize=(18, 12))
        
        ax1 = fig.add_subplot(231, projection='3d')
        user_movie_rating = pd.read_sql('''
            SELECT u.user_id, m.movie_id, r.rating, u.age, m.release_year
            FROM ratings r
            JOIN users u ON r.user_id = u.user_id
            JOIN movies m ON r.movie_id = m.movie_id
            LIMIT 200
        ''', conn)
        
        scatter1 = ax1.scatter(user_movie_rating['user_id'], 
                              user_movie_rating['movie_id'], 
                              user_movie_rating['rating'],
                              c=user_movie_rating['rating'], 
                              cmap='viridis',
                              s=20, alpha=0.6)
        ax1.set_xlabel('User ID')
        ax1.set_ylabel('Movie ID')
        ax1.set_zlabel('Rating')
        ax1.set_title('3D: Користувачі vs Фільми vs Рейтинги')
        plt.colorbar(scatter1, ax=ax1, shrink=0.6, label='Рейтинг')
        
        ax2 = fig.add_subplot(232, projection='3d')
        age_year_rating = pd.read_sql('''
            SELECT u.age, m.release_year, AVG(r.rating) as avg_rating
            FROM ratings r
            JOIN users u ON r.user_id = u.user_id
            JOIN movies m ON r.movie_id = m.movie_id
            WHERE u.age IS NOT NULL AND m.release_year IS NOT NULL
            GROUP BY u.age, m.release_year
        ''', conn)
        
        scatter2 = ax2.scatter(age_year_rating['age'], 
                              age_year_rating['release_year'], 
                              age_year_rating['avg_rating'],
                              c=age_year_rating['avg_rating'], 
                              cmap='plasma',
                              s=50, alpha=0.7)
        ax2.set_xlabel('Вік користувача')
        ax2.set_ylabel('Рік фільму')
        ax2.set_zlabel('Середній рейтинг')
        ax2.set_title('3D: Вік vs Рік фільму vs Рейтинг')
        plt.colorbar(scatter2, ax=ax2, shrink=0.6, label='Рейтинг')
        
        ax3 = fig.add_subplot(233, projection='3d')
        network_data = pd.read_sql('''
            SELECT fn.user_id1, fn.user_id2, u1.age as age1, u2.age as age2
            FROM friendster_network fn
            JOIN users u1 ON fn.user_id1 = u1.user_id
            JOIN users u2 ON fn.user_id2 = u2.user_id
            LIMIT 150
        ''', conn)
        
        scatter3 = ax3.scatter(network_data['user_id1'], 
                              network_data['user_id2'], 
                              network_data['age1'],
                              c=network_data['age2'], 
                              cmap='coolwarm',
                              s=30, alpha=0.6)
        ax3.set_xlabel('User ID 1')
        ax3.set_ylabel('User ID 2')
        ax3.set_zlabel('Вік User 1')
        ax3.set_title('3D: Соціальна мережа за віком')
        plt.colorbar(scatter3, ax=ax3, shrink=0.6, label='Вік User 2')
        
        ax4 = fig.add_subplot(234, projection='3d')
        genre_analysis = pd.read_sql('''
            SELECT 
                m.release_year,
                LENGTH(m.genres) as genre_complexity,
                COUNT(*) as rating_count,
                AVG(r.rating) as avg_rating
            FROM movies m
            JOIN ratings r ON m.movie_id = r.movie_id
            GROUP BY m.movie_id
            LIMIT 100
        ''', conn)
        
        scatter4 = ax4.scatter(genre_analysis['release_year'], 
                              genre_analysis['genre_complexity'], 
                              genre_analysis['avg_rating'],
                              c=genre_analysis['rating_count'], 
                              cmap='hot',
                              s=40, alpha=0.7)
        ax4.set_xlabel('Рік випуску')
        ax4.set_ylabel('Складність жанрів')
        ax4.set_zlabel('Середній рейтинг')
        ax4.set_title('3D: Рік vs Жанри vs Рейтинг')
        plt.colorbar(scatter4, ax=ax4, shrink=0.6, label='Кількість оцінок')
        
        ax5 = fig.add_subplot(235, projection='3d')
        time_analysis = pd.read_sql('''
            SELECT 
                u.user_id,
                COUNT(*) as activity_count,
                AVG(r.rating) as avg_rating,
                u.age
            FROM users u
            JOIN ratings r ON u.user_id = r.user_id
            GROUP BY u.user_id
            LIMIT 100
        ''', conn)
        
        scatter5 = ax5.scatter(time_analysis['user_id'], 
                              time_analysis['activity_count'], 
                              time_analysis['avg_rating'],
                              c=time_analysis['age'], 
                              cmap='spring',
                              s=35, alpha=0.7)
        ax5.set_xlabel('User ID')
        ax5.set_ylabel('Активність')
        ax5.set_zlabel('Середній рейтинг')
        ax5.set_title('3D: Активність vs Рейтинг vs Вік')
        plt.colorbar(scatter5, ax=ax5, shrink=0.6, label='Вік')
        
        ax6 = fig.add_subplot(236, projection='3d')
        behavior_analysis = pd.read_sql('''
            SELECT 
                ub.user_id,
                COUNT(*) as behavior_count,
                AVG(ub.duration_seconds) as avg_duration,
                AVG(r.rating) as avg_rating
            FROM user_behavior ub
            JOIN ratings r ON ub.user_id = r.user_id AND ub.movie_id = r.movie_id
            GROUP BY ub.user_id
            LIMIT 80
        ''', conn)
        
        scatter6 = ax6.scatter(behavior_analysis['user_id'], 
                              behavior_analysis['behavior_count'], 
                              behavior_analysis['avg_rating'],
                              c=behavior_analysis['avg_duration'], 
                              cmap='winter',
                              s=40, alpha=0.7)
        ax6.set_xlabel('User ID')
        ax6.set_ylabel('Кількість дій')
        ax6.set_zlabel('Середній рейтинг')
        ax6.set_title('3D: Поведінка vs Рейтинг vs Тривалість')
        plt.colorbar(scatter6, ax=ax6, shrink=0.6, label='Тривалість (сек)')
        
        plt.tight_layout()
        plt.savefig('media/3d_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_recommendation_queries(self):
        conn = sqlite3.connect(self.db_path)
        
        print("\n=== ПРИКЛАДИ РЕКОМЕНДАЦІЙНИХ ЗАПИТІВ ===")
        
        print("\n1. Топ фільмів за середнім рейтингом:")
        top_movies = pd.read_sql('''
            SELECT 
                m.title,
                m.genres,
                m.release_year,
                ROUND(AVG(r.rating), 2) as avg_rating,
                COUNT(*) as rating_count
            FROM movies m
            JOIN ratings r ON m.movie_id = r.movie_id
            GROUP BY m.movie_id, m.title, m.genres, m.release_year
            HAVING COUNT(*) >= 3
            ORDER BY avg_rating DESC
            LIMIT 10
        ''', conn)
        print(top_movies)
        
        print("\n2. Рекомендації на основі жанрів:")
        genre_recommendations = pd.read_sql('''
            SELECT 
                u.user_id,
                m.genres,
                COUNT(DISTINCT r.movie_id) as rated_movies,
                ROUND(AVG(r.rating), 2) as avg_rating
            FROM users u
            JOIN ratings r ON u.user_id = r.user_id
            JOIN movies m ON r.movie_id = m.movie_id
            GROUP BY u.user_id, m.genres
            ORDER BY u.user_id, rated_movies DESC
            LIMIT 15
        ''', conn)
        print(genre_recommendations)
        
        print("\n3. Соціальні рекомендації (друзі):")
        social_recommendations = pd.read_sql('''
            SELECT 
                u1.user_id as main_user,
                u2.user_id as friend_id,
                COUNT(DISTINCT r.movie_id) as common_movies,
                ROUND(AVG(r.rating), 2) as friend_avg_rating
            FROM friendster_network fn
            JOIN users u1 ON fn.user_id1 = u1.user_id
            JOIN users u2 ON fn.user_id2 = u2.user_id
            JOIN ratings r ON u2.user_id = r.user_id
            GROUP BY u1.user_id, u2.user_id
            HAVING common_movies >= 2
            ORDER BY main_user, common_movies DESC
            LIMIT 10
        ''', conn)
        print(social_recommendations)
        
        print("\n4. Демографічні рекомендації:")
        demographic_recommendations = pd.read_sql('''
            SELECT 
                u.gender,
                u.age,
                m.genres,
                COUNT(*) as rating_count,
                ROUND(AVG(r.rating), 2) as avg_rating
            FROM users u
            JOIN ratings r ON u.user_id = r.user_id
            JOIN movies m ON r.movie_id = m.movie_id
            GROUP BY u.gender, u.age, m.genres
            ORDER BY rating_count DESC
            LIMIT 10
        ''', conn)
        print(demographic_recommendations)
        
        conn.close()
        
    def run_complete_analysis(self):
        print("=== СИСТЕМА РЕКОМЕНДАЦІЙНИХ ФІЛЬМІВ ===\n")
        
        self.load_existing_data()
        self.create_database_schema()
        self.populate_database()
        self.analyze_database()
        self.create_visualizations()
        self.generate_recommendation_queries()
        
        print("\n=== АНАЛІЗ ЗАВЕРШЕНО ===")
        print(f"База даних збережена як: {self.db_path}")
        print("Візуалізації збережено в папці: media/")

def main():
    analyzer = MovieRecommendationDatabase()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
