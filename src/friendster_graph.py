import os
import sqlite3
import gzip
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommendationProject:

    def __init__(self, db_path: str = "movie_recommendation_system.db",
                 friendster_path: str = "com-friendster.ungraph.txt.gz") -> None:
        self.db_path = db_path
        self.friendster_path = friendster_path
        self.conn: sqlite3.Connection | None = None

        self.users: pd.DataFrame = pd.DataFrame()
        self.movies: pd.DataFrame = pd.DataFrame()
        self.ratings: pd.DataFrame = pd.DataFrame()

        self.user_item: pd.DataFrame = pd.DataFrame()
        self.user_similarity: pd.DataFrame = pd.DataFrame()

        self.media_dir = "media"
        os.makedirs(self.media_dir, exist_ok=True)


    def connect(self) -> None:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Файл бази даних не знайдено: {self.db_path}")

        self.conn = sqlite3.connect(self.db_path)
        print(f"Підключено до бази даних: {self.db_path}")

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            print("З’єднання з базою даних закрито.")

    def load_tables(self) -> None:
        if self.conn is None:
            raise RuntimeError("Спочатку виконайте connect().")

        print("Завантаження таблиць users, movies, ratings...")
        self.users = pd.read_sql("SELECT * FROM users", self.conn)
        self.movies = pd.read_sql("SELECT * FROM movies", self.conn)
        self.ratings = pd.read_sql("SELECT * FROM ratings", self.conn)

        print(f"Користувачів: {len(self.users)}")
        print(f"Фільмів: {len(self.movies)}")
        print(f"Оцінок: {len(self.ratings)}\n")


    def print_basic_info(self) -> None:
        print("ЗАГАЛЬНА ІНФОРМАЦІЯ ПРО НАБІР ДАНИХ")

        print("Перші 5 користувачів:")
        print(self.users.head())
        print("\nПерші 5 фільмів:")
        print(self.movies.head())
        print("\nПерші 5 оцінок:")
        print(self.ratings.head())

    def plot_basic_charts(self) -> None:
        plt.figure(figsize=(7, 4))
        self.ratings["rating"].hist(bins=10, edgecolor="black")
        plt.title("Розподіл оцінок фільмів")
        plt.xlabel("Оцінка")
        plt.ylabel("Кількість")
        plt.tight_layout()
        fname1 = os.path.join(self.media_dir, "hist_ratings.png")
        plt.savefig(fname1)
        print(f"Зображення збережено у файлі: {fname1}")
        plt.show()

        movie_counts = (self.ratings.groupby("movie_id")["rating"]
                        .count()
                        .sort_values(ascending=False)
                        .head(10))
        movie_titles = (self.movies.set_index("movie_id")
                        .loc[movie_counts.index, "title"])

        plt.figure(figsize=(9, 5))
        plt.barh(movie_titles, movie_counts.values)
        plt.gca().invert_yaxis()
        plt.title("Топ-10 найчастіше оцінених фільмів")
        plt.xlabel("Кількість оцінок")
        plt.tight_layout()
        fname2 = os.path.join(self.media_dir, "top_movies.png")
        plt.savefig(fname2)
        print(f"Зображення збережено у файлі: {fname2}")
        plt.show()

        user_counts = self.ratings["user_id"].value_counts()

        plt.figure(figsize=(7, 4))
        user_counts.hist(bins=20, edgecolor="black")
        plt.title("Розподіл активності користувачів (кількість оцінок)")
        plt.xlabel("Кількість оцінок")
        plt.ylabel("Кількість користувачів")
        plt.tight_layout()
        fname3 = os.path.join(self.media_dir, "user_activity.png")
        plt.savefig(fname3)
        print(f"Зображення збережено у файлі: {fname3}")
        plt.show()

    def build_user_item_matrix(self, min_ratings: int = 1) -> None:
        print("\nПобудова матриці 'користувач – фільм'...")

        if min_ratings > 1:
            activity = self.ratings["user_id"].value_counts()
            active_users = activity[activity >= min_ratings].index
            ratings_filtered = self.ratings[self.ratings["user_id"].isin(active_users)]
        else:
            ratings_filtered = self.ratings

        self.user_item = ratings_filtered.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        )

        print(f"Розмір матриці: {self.user_item.shape[0]} користувачів x "
              f"{self.user_item.shape[1]} фільмів (після фільтрації).")

    def compute_user_similarity(self) -> None:
        print("Обчислення матриці схожості користувачів...")

        if self.user_item.empty:
            raise RuntimeError("user_item порожня. Спочатку виконайте build_user_item_matrix().")

        filled = self.user_item.fillna(0.0)
        sim = cosine_similarity(filled.values)

        self.user_similarity = pd.DataFrame(
            sim,
            index=filled.index,
            columns=filled.index
        )

        print("Матриця схожості користувачів сформована.\n")

    def recommend_for_user(self,
                           user_id: int,
                           k_neighbors: int = 10,
                           n_recs: int = 5) -> List[Tuple[int, str, float]]:
        
        print(f"Формування рекомендацій для користувача {user_id}...")

        if self.user_similarity.empty or self.user_item.empty:
            raise RuntimeError("Спочатку побудуйте матрицю та обчисліть схожість користувачів.")

        if user_id not in self.user_item.index:
            print("Користувач відсутній у матриці user_item.")
            return []

        user_ratings = self.user_item.loc[user_id]
        rated_movies = user_ratings[user_ratings.notna()].index

        similarities = self.user_similarity.loc[user_id].drop(user_id)
        top_neighbors = similarities.sort_values(ascending=False).head(k_neighbors)

        scores = {}
        for movie_id in self.user_item.columns:
            if movie_id in rated_movies:
                continue

            num = 0.0
            den = 0.0
            for neighbor_id, sim in top_neighbors.items():
                rating = self.user_item.loc[neighbor_id, movie_id]
                if not np.isnan(rating):
                    num += sim * rating
                    den += sim

            if den > 0:
                scores[movie_id] = num / den

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]

        recs: List[Tuple[int, str, float]] = []
        movie_titles = self.movies.set_index("movie_id")["title"].to_dict()

        print("Отримані рекомендації:")
        for movie_id, score in sorted_scores:
            title = movie_titles.get(movie_id, f"Movie {movie_id}")
            print(f"{movie_id:4d} | {title:<50} | прогноз = {score:.2f}")
            recs.append((movie_id, title, float(score)))

        print()
        return recs

    def analyze_friendster_graph(self,
                                 node_limit: int = 1000,
                                 edge_limit: int = 8000) -> None:
        """
        Побудова підграфа Friendster та базова візуалізація.
        Графік і гістограма ступенів зберігаються в media/
        і відображаються на екрані.
        """
        print("Побудова підграфа Friendster...")

        if not os.path.exists(self.friendster_path):
            print(f"Файл графа Friendster не знайдено: {self.friendster_path}")
            return

        G = nx.Graph()

        with gzip.open(self.friendster_path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                try:
                    u, v = int(parts[0]), int(parts[1])
                except ValueError:
                    continue

                G.add_edge(u, v)
                if G.number_of_nodes() >= node_limit and G.number_of_edges() >= edge_limit:
                    break

        print(f"У підграф додано {G.number_of_nodes()} вузлів і {G.number_of_edges()} ребер.")

        degrees = [d for _, d in G.degree()]

        plt.figure(figsize=(7, 4))
        plt.hist(degrees, bins=40, edgecolor="black")
        plt.title("Розподіл ступенів вузлів у підграфі Friendster")
        plt.xlabel("Кількість зв’язків")
        plt.ylabel("Кількість вузлів")
        plt.tight_layout()
        fname_deg = os.path.join(self.media_dir, "friendster_degree_distribution.png")
        plt.savefig(fname_deg)
        print(f"Зображення збережено у файлі: {fname_deg}")
        plt.show()

        sub_nodes = list(G.nodes())[:80]
        subgraph = G.subgraph(sub_nodes)

        plt.figure(figsize=(7, 6))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(subgraph, pos,
                node_size=30,
                node_color="orange",
                edge_color="gray",
                with_labels=False)
        plt.title("Фрагмент соціального графа Friendster")
        fname_g = os.path.join(self.media_dir, "friendster_graph.png")
        plt.savefig(fname_g)
        print(f"Зображення збережено у файлі: {fname_g}")
        plt.show()

if __name__ == "__main__":
    project = MovieRecommendationProject(
        db_path="movie_recommendation_system.db",
        friendster_path="com-friendster.ungraph.txt.gz"
    )

    try:
        project.connect()
        project.load_tables()
        project.print_basic_info()
        project.plot_basic_charts()
        project.build_user_item_matrix(min_ratings=1)
        project.compute_user_similarity()
        project.recommend_for_user(user_id=1, k_neighbors=10, n_recs=5)
        project.analyze_friendster_graph(node_limit=1000, edge_limit=8000)
    finally:
        project.close()

    print("\nРоботу завершено. Усі зображення збережено у теці 'media'.")
