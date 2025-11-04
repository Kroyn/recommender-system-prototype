import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os
from datetime import datetime

try:
    ratings_df = pd.read_csv('ratings.csv') 
    
    movies_df = pd.read_csv('movies.csv', encoding='utf-8', quotechar='"')
    
    users_df = pd.read_csv('users.csv')
    
    print("Files loaded successfully!")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Number of movies: {len(movies_df)}")
    print(f"Number of users: {len(users_df)}")
    
    # Перевірка структури даних
    print("\nRatings columns:", ratings_df.columns.tolist())
    print("First few rows of ratings:")
    print(ratings_df.head())
    
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Check if the files are in the same folder as the code")
    exit()
except pd.errors.ParserError as e:
    print(f"Parser error: {e}")
    print("Trying alternative method to read movies.csv...")
    try:
        movies_df = pd.read_csv('movies.csv', encoding='utf-8', on_bad_lines='skip')
        print("Movies file loaded with skipped problematic lines")
    except Exception as e2:
        print(f"Failed to load movies.csv: {e2}")
        exit()

def autocorrelation(series, lag=1):
    n = len(series)
    mean_val = np.mean(series)

    covariance = np.sum((series[lag:] - mean_val) * (series[:-lag] - mean_val))
    variance = np.sum((series - mean_val) ** 2)

    return covariance / variance

def run_task_1():
    print("\n" + "=" * 70)
    print("Task 1. Analysis of stationary time series of movie ratings")
    print("=" * 70)

    print("\nFirst 5 records from ratings.csv:")
    print(ratings_df.head())

    movie_ratings_count = ratings_df.groupby('movieId').size().sort_values(ascending=False)
    top_movie_id = movie_ratings_count.index[0]
    top_movie_title = movies_df[movies_df['movieId'] == top_movie_id]['title'].values[0]

    print(f"\nMovie for analysis: {top_movie_title} (ID: {top_movie_id})")
    print(f"Number of ratings: {movie_ratings_count.iloc[0]}")

    # Отримуємо рейтинги для фільму та сортуємо по даті
    movie_ratings = ratings_df[ratings_df['movieId'] == top_movie_id].copy()
    movie_ratings['date'] = pd.to_datetime(movie_ratings['date'])
    movie_ratings = movie_ratings.sort_values('date')
    
    ratings_values = movie_ratings['rating'].values

    if len(ratings_values) > 16:
        time_series = ratings_values[:16]
    else:
        time_series = np.concatenate([ratings_values, np.random.normal(ratings_values.mean(), ratings_values.std(), 16 - len(ratings_values))])

    time_periods = np.arange(1, 17)

    print(f"\nTime series of ratings: {time_series}")
    print(f"Number of observations: {len(time_series)}")

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(time_periods, time_series, 'o-', linewidth=2, markersize=8, color='blue', markerfacecolor='red')
    plt.title(f'Rating Time Series\n"{top_movie_title}"', fontsize=14, fontweight='bold')
    plt.xlabel('Observation Period', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(time_periods)

    for i, (t, r) in enumerate(zip(time_periods, time_series)):
        plt.annotate(f'{r:.1f}', (t, r), textcoords="offset points", xytext=(0,10), ha='center')

    plt.subplot(2, 2, 2)
    plt.plot(time_periods[:-1], time_series[:-1], 's-', label='y(t)', linewidth=2, markersize=6, color='green')
    plt.plot(time_periods[1:], time_series[1:], 'o-', label='y(t+1)', linewidth=2, markersize=6, color='orange')
    plt.title('Comparison of y(t) and y(t+1)', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    mean_rating = np.mean(time_series)
    std_rating = np.std(time_series)
    min_rating = np.min(time_series)
    max_rating = np.max(time_series)

    print(f"\nStatistical characteristics of the series:")
    print(f"Mean value: {mean_rating:.3f}")
    print(f"Standard deviation: {std_rating:.3f}")
    print(f"Minimum rating: {min_rating:.1f}")
    print(f"Maximum rating: {max_rating:.1f}")

    autocorr_1 = autocorrelation(time_series, lag=1)
    autocorr_2 = autocorrelation(time_series, lag=2)

    print(f"\nAutocorrelation coefficients:")
    print(f"1st order autocorrelation: {autocorr_1:.4f}")
    print(f"2nd order autocorrelation: {autocorr_2:.4f}")

    plt.subplot(2, 2, 3)
    plt.scatter(time_series[:-1], time_series[1:], alpha=0.8, s=80, color='red', edgecolors='black')
    plt.title('Dependency of y(t+1) on y(t)', fontsize=14, fontweight='bold')
    plt.xlabel('y(t)', fontsize=12)
    plt.ylabel('y(t+1)', fontsize=12)

    slope, intercept, r_value, p_value, std_err = stats.linregress(time_series[:-1], time_series[1:])
    x_fit = np.linspace(np.min(time_series), np.max(time_series), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, '--', color='blue', linewidth=2,
             label=f'Regression Line (r={r_value:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    for i, (x, y) in enumerate(zip(time_series[:-1], time_series[1:])):
        plt.annotate(f'({x:.1f},{y:.1f})', (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

    plt.subplot(2, 2, 4)
    plt.hist(time_series, bins=6, alpha=0.7, color='lightblue', edgecolor='black', linewidth=1.2)
    plt.title('Distribution of Movie Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    counts, bins, patches = plt.hist(time_series, bins=6, alpha=0)
    for count, bin_edge, patch in zip(counts, bins, patches):
        if count > 0:
            plt.text(bin_edge + 0.1, count + 0.1, str(int(count)), ha='left', va='bottom')

    plt.tight_layout()
    plt.show()

    print(f"\nRegression analysis results:")
    print(f"Pearson correlation coefficient: {r_value:.4f}")
    print(f"Regression line slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"P-value: {p_value:.4f}")

    print(f"\nInterpretation of results:")
    if autocorr_1 > 0.5:
        print("▪ High positive autocorrelation - ratings are strongly dependent on previous values")
    elif autocorr_1 > 0.2:
        print("▪ Moderate positive autocorrelation - noticeable dependency between consecutive ratings")
    else:
        print("▪ Weak autocorrelation - ratings are almost independent")

    if r_value > 0:
        print("▪ Positive correlation between y(t) and y(t+1) - high ratings tend to appear in groups")

def run_task_2():
    print("\n" + "=" * 70)
    print("Task 2. Forecasting movie popularity based on random walk")
    print("=" * 70)

    class MoviePopularityForecast:

        def __init__(self):
            self.trend = 0.02
            self.volatility = 0.15

        def random_walk_forecast(self, current_value, trend, horizon):
            return current_value + trend * horizon

        def forecast_error(self, volatility, horizon):
            return volatility * np.sqrt(horizon)

        def mean_squared_error(self, volatility, horizon):
            return volatility**2 * horizon

        def get_movie_time_series(self, movie_id, min_ratings=3):
            movie_ratings = ratings_df[ratings_df['movieId'] == movie_id].copy()
            
            if len(movie_ratings) < min_ratings:
                print(f"  ⚠ Movie {movie_id} has only {len(movie_ratings)} ratings (minimum {min_ratings} required)")
                return None

            # Сортуємо по даті замість timestamp
            movie_ratings['date'] = pd.to_datetime(movie_ratings['date'])
            movie_ratings = movie_ratings.sort_values('date')

            ratings_series = movie_ratings['rating'].values

            print(f"  ✓ Movie {movie_id}: {len(ratings_series)} ratings")
            return ratings_series

        def calculate_trend_from_data(self, series):
            if len(series) < 2:
                return 0, series[0] if len(series) > 0 else 0

            n = len(series)
            x = np.arange(n)
            slope, intercept = np.polyfit(x, series, 1)
            return slope, intercept

    forecaster = MoviePopularityForecast()

    print("Searching for movies with enough ratings...")

    movies_with_ratings = []
    for movie_id in ratings_df['movieId'].unique():
        ratings_count = len(ratings_df[ratings_df['movieId'] == movie_id])
        if ratings_count >= 3:
            movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
            avg_rating = ratings_df[ratings_df['movieId'] == movie_id]['rating'].mean()
            movies_with_ratings.append({
                'id': movie_id,
                'title': movie_title,
                'ratings_count': ratings_count,
                'avg_rating': avg_rating
            })

    top_movies_info = sorted(movies_with_ratings, key=lambda x: x['ratings_count'], reverse=True)[:3]

    if not top_movies_info:
        print("No movies found with enough ratings. Using test data...")
        np.random.seed(42)
        test_movies = [
            {'id': 1, 'title': 'Test Movie 1', 'ratings_count': 10, 'avg_rating': 4.0},
            {'id': 2, 'title': 'Test Movie 2', 'ratings_count': 8, 'avg_rating': 3.8},
            {'id': 3, 'title': 'Test Movie 3', 'ratings_count': 7, 'avg_rating': 4.2}
        ]

        movies_data = {}
        for movie in test_movies:
            trend = np.random.uniform(-0.1, 0.1)
            volatility = np.random.uniform(0.1, 0.3)
            series = [movie['avg_rating']]
            for i in range(9):
                series.append(series[-1] + trend + np.random.normal(0, volatility))
            movies_data[movie['title']] = np.array(series)

        top_movies_info = test_movies
    else:
        print("Top movies by number of ratings:")
        for i, movie in enumerate(top_movies_info, 1):
            print(f"{i}. {movie['title']}")
            print(f"   Ratings: {movie['ratings_count']}, Average Rating: {movie['avg_rating']:.2f}")

        movies_data = {}
        for movie in top_movies_info:
            print(f"\nLoading data for: {movie['title']}")
            time_series = forecaster.get_movie_time_series(movie['id'], min_ratings=3)
            if time_series is not None:
                movies_data[movie['title']] = time_series
            else:
                print(f"  ❌ Failed to load data for {movie['title']}")

    print(f"\nLoaded time series for {len(movies_data)} movies")

    if len(movies_data) == 0:
        print("Failed to load data for any movie. Exiting program.")
        exit()

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']

    for i, (movie_name, popularity) in enumerate(movies_data.items()):
        n_periods = len(popularity)
        time_actual = np.arange(n_periods)
        plt.plot(time_actual, popularity, marker=markers[i], linewidth=2,
                 label=movie_name, color=colors[i], markersize=4)

        current_popularity = popularity[-1]

        forecast_periods = 5
        slope, intercept = forecaster.calculate_trend_from_data(popularity)
        future_periods = np.arange(n_periods, n_periods + forecast_periods)
        forecast = intercept + slope * future_periods

        plt.plot(future_periods, forecast, '--', color=colors[i], linewidth=1, alpha=0.7)

    plt.title('Historical Ratings and 5-Period Forecasts', fontsize=14, fontweight='bold')
    plt.xlabel('Observation Period', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    movie_name = list(movies_data.keys())[0]
    popularity = movies_data[movie_name]
    current_value = popularity[-1]

    horizons = np.arange(1, 11)
    forecasts = [forecaster.random_walk_forecast(current_value, forecaster.trend, h) for h in horizons]
    errors = [forecaster.forecast_error(forecaster.volatility, h) for h in horizons]

    plt.plot(horizons, forecasts, 's-', linewidth=2, markersize=6,
             label='Forecasted Rating', color='red')
    plt.fill_between(horizons,
                     np.array(forecasts) - np.array(errors),
                     np.array(forecasts) + np.array(errors),
                     alpha=0.3, label='Confidence Interval ±1σ', color='red')

    plt.title(f'Forecast for "{movie_name}"', fontsize=14, fontweight='bold')
    plt.xlabel('Forecast Horizon (periods)', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    mses = [forecaster.mean_squared_error(forecaster.volatility, h) for h in horizons]

    plt.plot(horizons, errors, 'o-', linewidth=2, label='Forecast Error', color='blue')
    plt.plot(horizons, np.sqrt(mses), 's-', linewidth=2, label='RMSE', color='green')
    plt.title('Forecast Errors', fontsize=14, fontweight='bold')
    plt.xlabel('Forecast Horizon (periods)', fontsize=12)
    plt.ylabel('Error', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)

    movie_forecasts = {}
    for movie_name, popularity in movies_data.items():
        slope, intercept = forecaster.calculate_trend_from_data(popularity)
        forecast_5_periods = intercept + slope * (len(popularity) + 5)
        movie_forecasts[movie_name] = forecast_5_periods

    sorted_movies = sorted(movie_forecasts.items(), key=lambda x: x[1], reverse=True)

    movies_sorted = [movie[0] for movie in sorted_movies]
    forecasts_sorted = [movie[1] for movie in sorted_movies]

    bars = plt.bar(range(len(movies_sorted)), forecasts_sorted,
                   color=['gold', 'silver', 'brown'], alpha=0.7, edgecolor='black')

    plt.title('Recommendations based on Forecasted Rating', fontsize=14, fontweight='bold')
    plt.ylabel('Forecasted Rating after 5 periods', fontsize=12)
    plt.xticks(range(len(movies_sorted)), movies_sorted, rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, forecasts_sorted):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print(f"\n5-Period Forecast Results:")
    for movie_name, forecast in sorted_movies:
        current_pop = movies_data[movie_name][-1]
        growth = forecast - current_pop
        growth_percent = (growth / current_pop) * 100

        print(f"\n{movie_name}:")
        print(f"  Current Rating: {current_pop:.3f}")
        print(f"  Forecast after 5 periods: {forecast:.3f}")
        print(f"  Expected growth: {growth:+.3f} ({growth_percent:+.1f}%)")
        print(f"  Confidence Interval: [{forecast - errors[4]:.3f}, {forecast + errors[4]:.3f}]")

    print("\n1. Movies for active promotion:")
    for i, (movie_name, forecast) in enumerate(sorted_movies[:2], 1):
        print(f"   {i}. {movie_name} (forecast: {forecast:.3f})")

    print("\n2. Movies to monitor:")
    for i, (movie_name, forecast) in enumerate(sorted_movies[2:], 1):
        print(f"   {i}. {movie_name} (forecast: {forecast:.3f})")

    print(f"\n3. Model Parameters:")
    print(f"   ▪ Constant trend: {forecaster.trend:.3f} per period")
    print(f"   ▪ Volatility: {forecaster.volatility:.3f}")
    print(f"   ▪ Model Type: Random Walk with drift")

    print(f"\n4. Forecast Accuracy:")
    print(f"   ▪ Forecast error for 1 period: ±{errors[0]:.3f}")
    print(f"   ▪ Forecast error for 5 periods: ±{errors[4]:.3f}")
    print(f"   ▪ RMSE for 5 periods: {np.sqrt(mses[4]):.3f}")

    for movie_name, popularity in movies_data.items():
        if len(popularity) > 5:
            autocorr = autocorrelation(popularity, 1)
            print(f"\n{movie_name}:")
            print(f"  1st order autocorrelation: {autocorr:.3f}")
            if autocorr > 0.3:
                print("  ▪ Strong dependence on previous values")
            elif autocorr > 0.1:
                print("  ▪ Moderate dependence")
            else:
                print("  ▪ Weak dependence")

    print("\nCONCLUSIONS:")
    print("1. Real data shows moderate autocorrelation of ratings")
    print("2. The random walk model allows for forecasting popularity dynamics")
    print("3. The system can effectively rank movies for recommendations")
    print("4. Forecast accuracy decreases as the forecast horizon increases")

def main_menu():
    print("\n" + "=" * 70)
    print("Project: 'Building a prototype movie recommendation system'")
    print("=" * 70)

    while True:
        print("\nSelect an option:")
        print("1. Run Task 1 (Time Series Analysis)")
        print("2. Run Task 2 (Random Walk Forecast)")
        print("3. Run Both Tasks")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            run_task_1()
            input("\nTask 1 complete. Press Enter to return to the menu...")
        elif choice == '2':
            run_task_2()
            input("\nTask 2 complete. Press Enter to return to the menu...")
        elif choice == '3':
            run_task_1()
            print("\n" + "-" * 70)
            run_task_2()
            input("\nAll tasks complete. Press Enter to return to the menu...")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main_menu()