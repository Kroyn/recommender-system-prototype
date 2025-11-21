import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import os

IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_data(db_path="movie_recommendation_system.db"):
    conn = sqlite3.connect(db_path)
    users = pd.read_sql("SELECT * FROM users", conn)
    movies = pd.read_sql("SELECT * FROM movies", conn)
    ratings = pd.read_sql("SELECT * FROM ratings", conn)
    conn.close()

    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5)]

    print("[INFO] Розміри набору даних:")
    print(f"  Користувачів: {len(users)}")
    print(f"  Фільмів:      {len(movies)}")
    print(f"  Оцінок:       {len(ratings)}\n")
    return users, movies, ratings


def eda(users, movies, ratings):
    print("[EDA] Приклади рядків таблиць:\n")
    print("users:\n", users.head(), "\n")
    print("movies:\n", movies.head(), "\n")
    print("ratings:\n", ratings.head(), "\n")

    for name, df in [("users", users), ("movies", movies), ("ratings", ratings)]:
        print(f"[EDA] Пропущені значення у {name}:")
        print(df.isnull().sum(), "\n")


def plot_basic_charts(ratings, movies):
    print("[CHARTS] Побудова аналітичних графіків...")

    plt.figure(figsize=(9, 5))
    plt.hist(ratings["rating"], bins=np.arange(0.5, 5.6, 0.5), edgecolor="black", alpha=0.7)
    plt.title("Розподіл оцінок фільмів")
    plt.xlabel("Оцінка")
    plt.ylabel("Кількість")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "hist_ratings.png"), dpi=300)
    plt.show()

    top_movies_counts = ratings["movie_id"].value_counts().head(10)
    top_movies = (
        pd.DataFrame({"movie_id": top_movies_counts.index, "count": top_movies_counts.values})
        .merge(movies, on="movie_id")
        .sort_values("count", ascending=True)
    )
    plt.figure(figsize=(10, 5))
    plt.barh(top_movies["title"], top_movies["count"])
    plt.title("Топ-10 найчастіше оцінених фільмів")
    plt.xlabel("Кількість оцінок")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "top_movies.png"), dpi=300)
    plt.show()

    genre_counts = {}
    for _, row in movies.iterrows():
        genres = str(row["genres"]).split("|")
        movie_id = row["movie_id"]
        movie_ratings_cnt = len(ratings[ratings["movie_id"] == movie_id])
        for g in genres:
            g = g.strip()
            if not g:
                continue
            genre_counts[g] = genre_counts.get(g, 0) + movie_ratings_cnt

    top_genres = pd.Series(genre_counts).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_genres.index, top_genres.values, color="#5b9bd5")
    plt.title("Топ-10 жанрів за кількістю оцінок")
    plt.xlabel("Кількість оцінок")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "genre_popularity.png"), dpi=300)
    plt.show()

    movies_with_ratings = ratings.merge(movies, on="movie_id")
    year_avg = movies_with_ratings.groupby("release_year")["rating"].mean().sort_index()

    plt.figure(figsize=(9, 5))
    plt.plot(year_avg.index, year_avg.values, marker="o", label="Середня оцінка", color="#4472c4")
    z = np.polyfit(year_avg.index, year_avg.values, 1)
    p = np.poly1d(z)
    plt.plot(year_avg.index, p(year_avg.index), "--", color="orange", label="Тренд")
    plt.title("Середня оцінка фільмів за роками випуску")
    plt.xlabel("Рік випуску")
    plt.ylabel("Середня оцінка")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "yearly_avg_rating.png"), dpi=300)
    plt.show()

    print("[IMG] Збережено: hist_ratings.png, top_movies.png, genre_popularity.png, yearly_avg_rating.png\n")


def build_movie_features(movies, verbose=True):
    genres_split = movies["genres"].str.get_dummies("|")
    movie_features = pd.concat([movies[["movie_id", "release_year"]], genres_split], axis=1)
    movie_features = movie_features.set_index("movie_id")
    if verbose:
        print("[FEATURES] Інженерія ознак виконана:")
        print(f"  Кількість жанрових ознак: {genres_split.shape[1]}")
        print(f"  Розмір матриці ознак: {movie_features.shape}\n")
    return movie_features


def plot_movie_similarity_heatmap(movie_features, movies, ratings, top_n=25):
    movie_counts = ratings.groupby("movie_id")["rating"].count().sort_values(ascending=False).head(top_n)
    movie_ids = movie_counts.index
    X = movie_features.loc[movie_ids]
    sim = cosine_similarity(X.values)
    titles = movies.set_index("movie_id").loc[movie_ids, "title"].apply(lambda x: (x[:25] + "…") if len(x) > 25 else x)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(sim, cmap="viridis")
    ax.set_xticks(np.arange(len(titles)))
    ax.set_yticks(np.arange(len(titles)))
    ax.set_xticklabels(titles, fontsize=7)
    ax.set_yticklabels(titles, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_title(f"Матриця подібності фільмів (ТОП-{top_n})", fontsize=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Подібність", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "movie_similarity_heatmap.png"), dpi=300)
    plt.show()


def precision_recall_at_k(predicted, relevant, k):
    if not predicted:
        return 0.0, 0.0
    k = min(k, len(predicted))
    predicted_at_k = predicted[:k]
    if not relevant:
        return 0.0, 0.0
    hit = len([m for m in predicted_at_k if m in relevant])
    precision = hit / k
    recall = hit / len(relevant)
    return precision, recall


def train_and_evaluate_models(ratings, movies, k_eval=10):
    print("[TRAIN] Навчання та оцінка моделей...\n")

    user_item = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
    user_item_filled = user_item.fillna(0)

    print(f"[MATRIX] Розмір матриці користувач–фільм: {user_item.shape[0]} користувачів x {user_item.shape[1]} фільмів\n")

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    print(f"[SPLIT] train={len(train)}, test={len(test)}\n")

    user_sim = cosine_similarity(user_item_filled)
    user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

    def predict_user_based(user_id, movie_id):
        if movie_id not in user_item.columns or user_id not in user_item.index:
            return ratings["rating"].mean()
        sims = user_sim_df.loc[user_id]
        col = user_item[movie_id]
        rated_mask = col.notna()
        if rated_mask.sum() == 0:
            return ratings["rating"].mean()
        numer = (sims[rated_mask] * col[rated_mask]).sum()
        denom = sims[rated_mask].sum()
        return numer / denom if denom != 0 else ratings["rating"].mean()

    preds, actuals = [], []
    for _, row in test.iterrows():
        preds.append(predict_user_based(row["user_id"], row["movie_id"]))
        actuals.append(row["rating"])
    rmse_cf = sqrt(mean_squared_error(actuals, preds))

    movie_features = build_movie_features(movies, verbose=False)
    sim_matrix = cosine_similarity(movie_features.values)
    sim_df = pd.DataFrame(sim_matrix, index=movie_features.index, columns=movie_features.index)

    prec_cf_list, rec_cf_list = [], []
    prec_cb_list, rec_cb_list = [], []

    users_in_test = test["user_id"].unique()
    for uid in users_in_test:
        test_u = test[test["user_id"] == uid]
        candidate_movies = test_u["movie_id"].unique()
        if len(candidate_movies) == 0:
            continue

        relevant = set(test_u.loc[test_u["rating"] >= 4.0, "movie_id"].unique())

        if uid in user_item.index:
            scores_cf = {}
            for mid in candidate_movies:
                scores_cf[mid] = predict_user_based(uid, mid)
            sorted_cf = [mid for mid, _ in sorted(scores_cf.items(), key=lambda x: x[1], reverse=True)]
            p_cf, r_cf = precision_recall_at_k(sorted_cf, relevant, k_eval)
            if relevant:
                prec_cf_list.append(p_cf)
                rec_cf_list.append(r_cf)

        liked_train = train[(train["user_id"] == uid) & (train["rating"] >= 4.0)]["movie_id"].unique()
        liked_train = [m for m in liked_train if m in sim_df.index]
        if liked_train:
            scores_cb = {}
            for mid in candidate_movies:
                if mid not in sim_df.index:
                    continue
                sims = sim_df.loc[liked_train, mid]
                scores_cb[mid] = sims.mean() if not sims.isna().all() else 0.0
            if scores_cb:
                sorted_cb = [mid for mid, _ in sorted(scores_cb.items(), key=lambda x: x[1], reverse=True)]
                p_cb, r_cb = precision_recall_at_k(sorted_cb, relevant, k_eval)
                if relevant:
                    prec_cb_list.append(p_cb)
                    rec_cb_list.append(r_cb)

    mean_prec_cf = np.mean(prec_cf_list) if prec_cf_list else 0.0
    mean_rec_cf = np.mean(rec_cf_list) if rec_cf_list else 0.0
    mean_prec_cb = np.mean(prec_cb_list) if prec_cb_list else 0.0
    mean_rec_cb = np.mean(rec_cb_list) if rec_cb_list else 0.0

    print("[METRICS] Оцінка якості моделей (test):")
    print(f"  User-based CF : RMSE = {rmse_cf:.4f}, Precision@{k_eval:.0f} = {mean_prec_cf:.4f}, Recall@{k_eval:.0f} = {mean_rec_cf:.4f}")
    print(f"  Content-based : Precision@{k_eval:.0f} = {mean_prec_cb:.4f}, Recall@{k_eval:.0f} = {mean_rec_cb:.4f}\n")

    return user_item_filled, user_sim_df, movie_features, sim_df


def get_recommendations(user_item_filled, user_sim_df, movie_features, sim_df, movies, ratings):
    user_id = int(input("Введіть ID користувача: "))
    if user_id not in user_item_filled.index:
        print("Такого користувача немає у базі.\n")
        return

    user_ratings_row = user_item_filled.loc[user_id]
    unrated = user_ratings_row[user_ratings_row == 0]
    sims = user_sim_df.loc[user_id]

    cf_scores = {}
    for movie_id in unrated.index:
        col = user_item_filled[movie_id]
        numer = (sims * col).sum()
        denom = sims.sum()
        cf_scores[movie_id] = numer / denom if denom != 0 else 0.0

    cf_top = sorted(cf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    liked_movies = ratings[(ratings["user_id"] == user_id) & (ratings["rating"] >= 4.0)]["movie_id"].unique()
    liked_movies = [m for m in liked_movies if m in sim_df.index]

    cb_scores = {}
    if liked_movies:
        for mid in movie_features.index:
            if mid in liked_movies:
                continue
            sims_mid = sim_df.loc[liked_movies, mid]
            cb_scores[mid] = sims_mid.mean() if not sims_mid.isna().all() else 0.0

    cb_top = sorted(cb_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print(f"\nUser-based CF рекомендації для користувача {user_id}:\n")
    for i, (mid, score) in enumerate(cf_top, 1):
        title = movies.loc[movies["movie_id"] == mid, "title"].values[0]
        print(f"{i:2}. {title:50} (прогноз = {score:.2f})")

    print(f"\nContent-based рекомендації для користувача {user_id}:\n")
    for i, (mid, score) in enumerate(cb_top, 1):
        title = movies.loc[movies["movie_id"] == mid, "title"].values[0]
        print(f"{i:2}. {title:50} (схожість = {score:.3f})")

    with open(os.path.join(IMAGES_DIR, "recommendations.txt"), "w", encoding="utf-8") as f:
        f.write(f"Рекомендації для користувача {user_id}\n\n")
        f.write("User-based CF:\n")
        for i, (mid, score) in enumerate(cf_top, 1):
            title = movies.loc[movies["movie_id"] == mid, "title"].values[0]
            f.write(f"{i}. {title} (прогноз = {score:.2f})\n")
        f.write("\nContent-based:\n")
        for i, (mid, score) in enumerate(cb_top, 1):
            title = movies.loc[movies["movie_id"] == mid, "title"].values[0]
            f.write(f"{i}. {title} (схожість = {score:.3f})\n")
    print("\n[OK] Рекомендації збережено у images/recommendations.txt\n")


def main():
    users, movies, ratings = load_data()
    user_item_filled = None
    user_sim_df = None
    movie_features = None
    sim_df = None

    while True:
        print("\nМеню:")
        print(" 1 - EDA та очищення даних")
        print(" 2 - Побудова графіків")
        print(" 3 - Інженерія ознак та heatmap")
        print(" 4 - Навчання й оцінка моделей")
        print(" 5 - Рекомендації")
        print(" 0 - Вихід")
        choice = input("Ваш вибір: ").strip()
        print()

        if choice == "1":
            eda(users, movies, ratings)
        elif choice == "2":
            plot_basic_charts(ratings, movies)
        elif choice == "3":
            movie_features_local = build_movie_features(movies, verbose=True)
            plot_movie_similarity_heatmap(movie_features_local, movies, ratings)
        elif choice == "4":
            user_item_filled, user_sim_df, movie_features, sim_df = train_and_evaluate_models(ratings, movies)
        elif choice == "5":
            if user_item_filled is None or user_sim_df is None or movie_features is None or sim_df is None:
                print("Спершу виконайте пункт 4, щоб навчити моделі та побудувати матриці подібності.\n")
            else:
                get_recommendations(user_item_filled, user_sim_df, movie_features, sim_df, movies, ratings)
        elif choice == "0":
            print("Роботу завершено.")
            break
        else:
            print("Невірний вибір, спробуйте ще раз.")


if __name__ == "__main__":
    main()