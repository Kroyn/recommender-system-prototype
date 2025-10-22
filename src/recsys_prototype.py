import os
import zipfile
import requests
import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from functools import lru_cache

ML100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "ml-100k"

def download_ml100k(dest_dir=DATA_DIR):
    if os.path.exists(dest_dir) and os.path.isdir(dest_dir):
        print(f"Dataset already exists in {dest_dir}")
        return
    print("Downloading MovieLens 100k...")
    r = requests.get(ML100K_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    print("Extraction complete.")

download_ml100k()

ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
ratings = pd.read_csv(
    os.path.join(DATA_DIR, "u.data"),
    sep="\t",
    names=ratings_cols,
    encoding='latin-1'
)

item_cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + \
            [f"g{i}" for i in range(19)]
items = pd.read_csv(
    os.path.join(DATA_DIR, "u.item"),
    sep="|",
    names=item_cols,
    encoding='latin-1'
)

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def make_genre_str(row):
    flags = [int(row[f"g{i}"]) for i in range(19)]
    genres = [gn for gn, fval in zip(GENRE_NAMES, flags) if fval == 1]
    return " ".join(genres) if genres else "unknown"

items["genres"] = items.apply(make_genre_str, axis=1)
items = items[["movie_id", "title", "genres"]]

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

popularity = train_df.groupby("item_id")["rating"].agg(["count", "mean"])
popularity = popularity.sort_values(by=["count", "mean"], ascending=False)

# Optimization: use LRU cache
@lru_cache(maxsize=128)
def recommend_popularity(k=10):
    top_items = popularity.reset_index().head(k)["item_id"].tolist()
    return items[items["movie_id"].isin(top_items)][["movie_id", "title"]].head(k)

tf = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'/-]+\b")
tfidf_matrix = tf.fit_transform(items["genres"])
item_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

movie_id_to_idx = {mid: idx for idx, mid in enumerate(items["movie_id"].tolist())}
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}

def recommend_content_based(user_id, k=10):
    user_hist = train_df[(train_df.user_id == user_id) & (train_df.rating >= 4)]
    if user_hist.empty:
        return recommend_popularity(k)

    sim_scores = np.zeros(item_sim.shape[0])
    for mid in user_hist.item_id:
        if mid in movie_id_to_idx:
            sim_scores += item_sim[movie_id_to_idx[mid]]

    rated = set(train_df[train_df.user_id == user_id].item_id.tolist())
    candidates = [
        (idx, score) for idx, score in enumerate(sim_scores)
        if items.iloc[idx]["movie_id"] not in rated
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    topk = [idx_to_movie_id[idx] for idx, _ in candidates[:k]]
    return items[items["movie_id"].isin(topk)][["movie_id", "title"]].head(k)

all_users = sorted(ratings.user_id.unique())
all_items = sorted(items.movie_id.unique())
user_to_idx = {u: i for i, u in enumerate(all_users)}
item_to_idx = {m: i for i, m in enumerate(all_items)}

rows = train_df.user_id.map(user_to_idx).to_list()
cols = train_df.item_id.map(item_to_idx).to_list()
data_vals = train_df.rating.to_list()
user_item_mat = csr_matrix(
    (data_vals, (rows, cols)),
    shape=(len(all_users), len(all_items))
)

item_item_sim = cosine_similarity(user_item_mat.T)

def predict_rating_item_based(user_id, item_id, k=20):
    if user_id not in user_to_idx or item_id not in item_to_idx:
        return None

    uidx = user_to_idx[user_id]
    iidx = item_to_idx[item_id]
    sims = item_item_sim[iidx]
    user_ratings = user_item_mat[uidx].toarray().ravel()
    rated_idx = np.where(user_ratings > 0)[0]

    if len(rated_idx) == 0:
        return None

    sim_and_ratings = [(sims[j], user_ratings[j]) for j in rated_idx]
    sim_and_ratings.sort(key=lambda x: x[0], reverse=True)
    top = sim_and_ratings[:k]

    num = sum(s * r for s, r in top)
    den = sum(abs(s) for s, _ in top)

    return num / den if den != 0 else None

def recommend_item_based(user_id, k=10):
    if user_id not in user_to_idx:
        return recommend_popularity(k)

    uidx = user_to_idx[user_id]
    user_ratings = user_item_mat[uidx].toarray().ravel()
    candidates = []

    for iidx, mid in enumerate(all_items):
        if user_ratings[iidx] == 0:
            pred = predict_rating_item_based(user_id, mid, k=20)
            if pred is not None:
                candidates.append((mid, pred))

    candidates.sort(key=lambda x: x[1], reverse=True)
    topk = [mid for mid, _ in candidates[:k]]
    return items[items["movie_id"].isin(topk)][["movie_id", "title"]].head(k)

def recommend_knapsack(user_id, k=10, time_budget=120):
    if user_id not in user_to_idx:
        return recommend_popularity(k)

    uidx = user_to_idx[user_id]
    user_ratings = user_item_mat[uidx].toarray().ravel()

    candidates = []
    for iidx, mid in enumerate(all_items):
        if user_ratings[iidx] == 0:
            pred = predict_rating_item_based(user_id, mid, k=20)
            if pred is not None:
                duration = np.random.randint(80, 181)
                candidates.append((mid, pred, duration))

    if not candidates:
        return recommend_popularity(k)

    n = len(candidates)
    dp = [[0] * (time_budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        mid, rating, duration = candidates[i - 1]
        for w in range(time_budget + 1):
            if duration <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - duration] + rating)
            else:
                dp[i][w] = dp[i - 1][w]

    selected = []
    w = time_budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(candidates[i - 1][0])
            w -= candidates[i - 1][2]

    selected.reverse()
    result = items[items["movie_id"].isin(selected[:k])][["movie_id", "title"]]
    return result.head(k)

def evaluate_item_based(test_df):
    preds, trues = [], []

    for _, row in test_df.iterrows():
        pred = predict_rating_item_based(row.user_id, row.item_id, k=20)
        if pred is not None:
            preds.append(pred)
            trues.append(row.rating)

    if len(preds) == 0:
        return None

    mse = mean_squared_error(trues, preds)
    return np.sqrt(mse)

def main_menu():
    while True:
        print("\n" + "="*50)
        print("  MOVIE RECOMMENDATION SYSTEM")
        print("="*50)
        print("1. Popularity Baseline")
        print("2. Content-Based Filtering (Genres)")
        print("3. Item-Based Collaborative Filtering")
        print("4. Knapsack-Based Recommendation (time-constrained)")
        print("5. Evaluate (RMSE for Item-Based)")
        print("0. Exit")
        print("="*50)

        choice = input("Select an option: ").strip()

        match choice:
            case "0":
                print("Goodbye!")
                break

            case "1":
                print("\nTOP-10 Most Popular Movies:")
                print(recommend_popularity(10).to_string(index=False))

            case "2":
                try:
                    uid = int(input("Enter user ID (1-943): "))
                    recs = recommend_content_based(uid, 10)
                    print(f"\nRecommendations for user {uid} (Content-Based):")
                    print(recs.to_string(index=False))
                except ValueError:
                    print("Invalid input. Please enter a number.")

            case "3":
                try:
                    uid = int(input("Enter user ID (1-943): "))
                    recs = recommend_item_based(uid, 10)
                    print(f"\nRecommendations for user {uid} (Item-Based CF):")
                    print(recs.to_string(index=False))
                except ValueError:
                    print("Invalid input. Please enter a number.")

            case "4":
                try:
                    uid = int(input("Enter user ID (1-943): "))
                    budget = int(input("Enter time budget in minutes (default 120): ") or "120")
                    recs = recommend_knapsack(uid, 10, budget)
                    print(f"\nRecommendations for user {uid} (Knapsack, {budget} min):")
                    print(recs.to_string(index=False))
                except ValueError:
                    print("Invalid input. Please enter valid numbers.")

            case "5":
                print("Computing RMSE (this may take 1-2 minutes)...")
                rmse_item = evaluate_item_based(test_df)
                if rmse_item:
                    print(f"Item-Based Collaborative Filtering RMSE: {rmse_item:.4f}")
                else:
                    print("Error: Unable to compute RMSE")

            case _:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
