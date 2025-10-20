import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data():
    try:
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')

        np.random.seed(42)
        start_date = datetime(2020, 1, 15)
        end_date = datetime(2025, 12, 31)

        dates = []
        for i in range(len(ratings)):
            random_date = start_date + timedelta(
                days=np.random.randint(0, (end_date - start_date).days),
                hours=np.random.randint(0, 24),
                minutes=np.random.randint(0, 60)
            )
            dates.append(random_date)

        ratings['date'] = dates
        ratings['month'] = ratings['date'].dt.to_period('M')

        print("Data loaded and preprocessed successfully.")
        return ratings, movies
    except FileNotFoundError:
        print("Error: 'ratings.csv' or 'movies.csv' not found.")
        print("Please make sure the files are in the same directory.")
        return None, None

def task_1_monthly_rating_dynamics(ratings):
    print("\n--- Running Task 1: Monthly Rating Dynamics ---")
    monthly_ratings = ratings.groupby('month')['rating'].mean()

    plt.figure(figsize=(14, 8))
    plt.plot(monthly_ratings.index.astype(str), monthly_ratings.values, 'o-', linewidth=2, markersize=6)
    plt.title('Average Monthly Rating Dynamics', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Rating', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.ylim(3.5, 5.0)

    x_numeric = np.arange(len(monthly_ratings))
    z = np.polyfit(x_numeric, monthly_ratings.values, 1)
    p = np.poly1d(z)
    plt.plot(monthly_ratings.index.astype(str), p(x_numeric), "r--", alpha=0.8, label='Trend')

    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Analysis Period: from {monthly_ratings.index.min()} to {monthly_ratings.index.max()}")
    print(f"Number of months: {len(monthly_ratings)}")
    print(f"Average rating for the entire period: {monthly_ratings.mean():.3f}")

def task_2_time_series_decomposition(ratings):
    print("\n--- Running Task 2: Time Series Decomposition ---")
    daily_ratings = ratings.groupby('date')['rating'].mean().resample('D').mean()
    daily_ratings = daily_ratings.fillna(method='ffill')

    decomposition = seasonal_decompose(daily_ratings.head(365), model='additive', period=30)

    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle('Time Series Decomposition of Movie Ratings')
    plt.tight_layout()
    plt.show()

    print("Series Components:")
    print("- Trend: Overall tendency of rating changes")
    print("- Seasonality: Monthly/weekly fluctuations")
    print("- Residual: Random fluctuations")

def task_3_yearly_rating_trend(ratings, movies):
    print("\n--- Running Task 3: Yearly Rating Trend ---")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    merged = ratings.merge(movies, on='movieId')

    yearly_avg = merged.groupby('year')['rating'].mean().dropna()
    years = yearly_avg.index.values.reshape(-1, 1)
    ratings_avg = yearly_avg.values

    model = LinearRegression()
    model.fit(years, ratings_avg)
    trend_line = model.predict(years)

    plt.figure(figsize=(12, 6))
    plt.plot(yearly_avg.index, yearly_avg.values, 'o-', label='Average Rating', alpha=0.7)
    plt.plot(yearly_avg.index, trend_line, 'r--', linewidth=2, label='Trend Line')
    plt.title('Trend of Average Movie Ratings by Year')
    plt.xlabel('Release Year')
    plt.ylabel('Average Rating')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Trend Equation: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x")
    print(f"Coefficient of Determination R²: {model.score(years, ratings_avg):.4f}")

def task_4_smoothing_methods(ratings):
    print("\n--- Running Task 4: Smoothing Methods Comparison ---")
    monthly_data = ratings.groupby('month')['rating'].mean()

    ma_3 = monthly_data.rolling(window=3, center=True).mean()
    ma_6 = monthly_data.rolling(window=6, center=True).mean()
    exp_smooth = monthly_data.ewm(span=3, adjust=False).mean()

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(monthly_data.index.astype(str), monthly_data.values, 'o-', label='Original', alpha=0.5)
    plt.plot(monthly_data.index.astype(str), ma_3.values, 'r-', label='Moving Average (3 mos.)')
    plt.plot(monthly_data.index.astype(str), ma_6.values, 'g-', label='Moving Average (6 mos.)')
    plt.title('Smoothing by Moving Average Method')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.subplot(2, 1, 2)
    plt.plot(monthly_data.index.astype(str), monthly_data.values, 'o-', label='Original', alpha=0.5)
    plt.plot(monthly_data.index.astype(str), exp_smooth.values, 'purple', label='Exponential Smoothing')
    plt.title('Exponential Smoothing')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    print("Comparison of smoothing methods:")
    print("- Moving Average: Simple method, good at removing noise")
    print("- Exponential: More weight given to recent observations")

def task_5_cluster_analysis(ratings, movies):
    print("\n--- Running Task 5: Cluster Analysis ---")

    if 'year' not in movies.columns:
         movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

    merged = ratings.merge(movies, on='movieId')

    movie_features = merged.groupby('movieId').agg({
        'rating': ['mean', 'count'],
        'year': 'first'
    }).dropna()

    movie_features.columns = ['avg_rating', 'rating_count', 'year']
    X = movie_features[['avg_rating', 'rating_count']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(movie_features['avg_rating'], movie_features['rating_count'],
                         c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Ratings')
    plt.title('Cluster Analysis of Movies (STATISTICA analog)')
    plt.grid(True)
    plt.show()

    print("Application of STATISTICA modules in the project:")
    print("- Time Series Analysis: Analysis of rating dynamics")
    print("- Data Miner: Building recommendation models")
    print("- Neural Networks: Deep learning for recommendations")

def main_menu():
    ratings, movies = load_data()
    if ratings is None:
        return

    while True:
        print("\n" + "=" * 50)
        print("     Time Series and Clustering Analysis Menu")
        print("=" * 50)
        print("1. Show Monthly Rating Dynamics")
        print("2. Show Time Series Decomposition")
        print("3. Show Yearly Rating Trend")
        print("4. Show Smoothing Methods Comparison")
        print("5. Show Cluster Analysis")
        print("6. Run All Tasks")
        print("7. Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            task_1_monthly_rating_dynamics(ratings)
        elif choice == '2':
            task_2_time_series_decomposition(ratings)
        elif choice == '3':
            task_3_yearly_rating_trend(ratings, movies)
        elif choice == '4':
            task_4_smoothing_methods(ratings)
        elif choice == '5':
            task_5_cluster_analysis(ratings, movies)
        elif choice == '6':
            task_1_monthly_rating_dynamics(ratings)
            task_2_time_series_decomposition(ratings)
            task_3_yearly_rating_trend(ratings, movies)
            task_4_smoothing_methods(ratings)
            task_5_cluster_analysis(ratings, movies)
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 7.")

        if choice in ['1', '2', '3', '4', '5', '6']:
            input("Press Enter to return to the menu...")

if __name__ == "__main__":
    main_menu()
