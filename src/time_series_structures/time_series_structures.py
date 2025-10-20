import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import os

def load_data():
    """Loads and preprocesses the data."""
    try:
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure 'ratings.csv' and 'movies.csv' are in the correct directory.")
        return None, None, None

    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    merged = ratings.merge(movies, on='movieId')

    return ratings, movies, merged

def prepare_yearly_stats(merged):
    """Aggregates stats by year."""
    yearly_stats = merged.groupby('year').agg({
        'rating': ['mean', 'count']
    }).round(3)
    yearly_stats.columns = ['avg_rating', 'rating_count']
    yearly_stats = yearly_stats.dropna()
    return yearly_stats

def task_1_plot_yearly_stats(yearly_stats):
    """Task 1: Plot average rating and rating count by year."""
    print("\n--- Task 1: Plotting Yearly Statistics ---")
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(yearly_stats.index, yearly_stats['avg_rating'], marker='o')
    plt.title('Average Movie Rating by Year')
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(yearly_stats.index, yearly_stats['rating_count'], marker='s', color='red')
    plt.title('Number of Ratings by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Ratings')
    plt.grid(True)

    plt.tight_layout()
    # plt.show() тепер блокує виконання, поки вікно не закриють
    plt.show()

def task_2_linear_regression(yearly_stats):
    """Task 2: Perform linear regression on rating vs. year."""
    print("\n--- Task 2: Linear Regression for Rating vs. Year ---")
    X = yearly_stats.index.values.reshape(-1, 1)
    y = yearly_stats['avg_rating'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)

    print(f'Coefficient of Determination R²: {r2:.4f}')
    print(f'Equation: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x')

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.title('Linear Regression: Rating vs. Year')
    plt.xlabel('Year')
    plt.ylabel('Average Rating')
    plt.legend()
    plt.grid(True)
    plt.show()
    return model

def task_3_forecast_future(yearly_stats, rating_model):
    """Task 3: Forecast future ratings and counts."""
    print("\n--- Task 3: Forecasting for 2024-2025 ---")

    future_years = np.array([2024, 2025]).reshape(-1, 1)
    future_ratings = rating_model.predict(future_years)

    print(f'Forecasted average rating for 2024: {future_ratings[0]:.3f}')
    print(f'Forecasted average rating for 2025: {future_ratings[1]:.3f}')

    X_count = yearly_stats.index.values.reshape(-1, 1)
    y_count = yearly_stats['rating_count'].values

    model_count = LinearRegression()
    model_count.fit(X_count, y_count)
    future_counts = model_count.predict(future_years)

    print(f'Forecasted number of ratings for 2024: {future_counts[0]:.0f}')
    print(f'Forecasted number of ratings for 2025: {future_counts[1]:.0f}')

def task_4_correlation_matrix(yearly_stats):
    """Task 4: Display correlation matrix."""
    print("\n--- Task 4: Correlation Matrix ---")
    corr = yearly_stats.corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f')
    plt.title('Correlation Matrix')
    plt.show()

def task_5_time_series_smoothing(yearly_stats):
    """Task 5: Apply Moving Average smoothing."""
    print("\n--- Task 5: Time Series Smoothing (Moving Average) ---")
    yearly_stats['rating_ma'] = yearly_stats['avg_rating'].rolling(window=3, center=True).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(yearly_stats.index, yearly_stats['avg_rating'], label='Original', alpha=0.7)
    plt.plot(yearly_stats.index, yearly_stats['rating_ma'], label='Smoothed (MA)', linewidth=2)
    plt.title('Smoothing the Average Rating Time Series')
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.legend()
    plt.grid(True)
    plt.show()

def task_6_age_audience_analysis(ratings, movies, merged):
    """Task 6: Analyze ratings for the 17-50 age group."""
    print("\n--- Task 6: Analysis for Age Audience (17-50) ---")

    np.random.seed(42)
    user_ages = pd.DataFrame({
        'userId': ratings['userId'].unique(),
        'age': np.random.randint(17, 51, size=ratings['userId'].nunique())
    })

    merged_age = ratings.merge(user_ages, on='userId').merge(movies, on='movieId')

    target_ratings = merged_age[(merged_age['age'] >= 17) & (merged_age['age'] <= 50)]

    age_avg_rating = target_ratings.groupby('age')['rating'].mean()

    plt.figure(figsize=(10, 5))
    age_avg_rating.plot(kind='bar', color='lightgreen')
    plt.title('Average Rating by User Age (17-50)')
    plt.xlabel('Age')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    top_movies = target_ratings.groupby('movieId')['rating'].mean().nlargest(10)
    top_movies_with_titles = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title']]
    top_movies_with_titles = top_movies_with_titles.merge(top_movies, on='movieId')
    print("\nTop 10 movies for audience 17-50:")
    print(top_movies_with_titles.round(3))

def main_menu():
    """Main menu to select tasks."""

    ratings, movies, merged = load_data()
    if ratings is None:
        return

    yearly_stats = prepare_yearly_stats(merged)
    rating_model = None # Ми ініціалізуємо модель тут

    while True:
        print("\n" + "=" * 50)
        print("     Time Series Analysis Menu")
        print("=" * 50)
        print("1. Plot Yearly Statistics (Rating and Count)")
        print("2. Run Linear Regression (Rating vs. Year)")
        print("3. Forecast Future Ratings (2024-2025)")
        print("4. Show Correlation Matrix")
        print("5. Show Time Series Smoothing (Moving Average)")
        print("6. Run Age Audience Analysis (17-50)")
        print("7. Run All Tasks")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        if choice == '1':
            task_1_plot_yearly_stats(yearly_stats)
            input("\nPlot closed. Press Enter to return to the menu...")

        elif choice == '2':
            # Зберігаємо навчену модель, щоб її можна було використати в завданні 3
            rating_model = task_2_linear_regression(yearly_stats)
            input("\nPlot closed. Press Enter to return to the menu...")

        elif choice == '3':
            if rating_model is None:
                print("\n[Warning] Please run Task 2 (Linear Regression) first to build the model.")
                # Опціонально: автоматично запускаємо завдання 2
                print("Running Task 2 now to build the model...")
                rating_model = task_2_linear_regression(yearly_stats)
                print("\nNow running Task 3...")
                task_3_forecast_future(yearly_stats, rating_model)
            else:
                task_3_forecast_future(yearly_stats, rating_model)
            input("\nTask complete. Press Enter to return to the menu...")

        elif choice == '4':
            task_4_correlation_matrix(yearly_stats)
            input("\nPlot closed. Press Enter to return to the menu...")

        elif choice == '5':
            task_5_time_series_smoothing(yearly_stats)
            input("\nPlot closed. Press Enter to return to the menu...")

        elif choice == '6':
            task_6_age_audience_analysis(ratings, movies, merged)
            input("\nPlot closed. Press Enter to return to the menu...")

        elif choice == '7':
            print("\n--- Running All Tasks ---")
            task_1_plot_yearly_stats(yearly_stats)
            print("Close the plot window to continue...")
            rating_model = task_2_linear_regression(yearly_stats)
            print("Close the plot window to continue...")
            task_3_forecast_future(yearly_stats, rating_model)
            task_4_correlation_matrix(yearly_stats)
            print("Close the plot window to continue...")
            task_5_time_series_smoothing(yearly_stats)
            print("Close the plot window to continue...")
            task_6_age_audience_analysis(ratings, movies, merged)
            print("\n--- All tasks completed. ---")
            input("\nPlots closed. Press Enter to return to the menu...")

        elif choice == '8':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number from 1 to 8.")

if __name__ == "__main__":
    main_menu()
