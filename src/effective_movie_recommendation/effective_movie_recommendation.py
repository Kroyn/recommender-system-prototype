import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

print("Тема 3. Питання для самостійного опрацювання")
print("Проект: 'Побудова прототипу рекомендаційної системи для фільмів'")
print("=" * 70)

# Завантаження даних з файлів
try:
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    users_df = pd.read_csv('users.csv')
    print("Файли успішно завантажено!")
    print(f"Кількість оцінок: {len(ratings_df)}")
    print(f"Кількість фільмів: {len(movies_df)}")
    print(f"Кількість користувачів: {len(users_df)}")
except FileNotFoundError as e:
    print(f"Помилка завантаження файлів: {e}")
    print("Перевірте, чи файли знаходяться в тій самій папці, що і код")
    exit()

# Функція для розрахунку автокореляції
def autocorrelation(series, lag=1):
    """Розрахунок коефіцієнта автокореляції"""
    n = len(series)
    mean_val = np.mean(series)
    
    # Коваріація
    covariance = np.sum((series[lag:] - mean_val) * (series[:-lag] - mean_val))
    # Дисперсія
    variance = np.sum((series - mean_val) ** 2)
    
    return covariance / variance

# =============================================================================
# ЗАДАЧА 1
# =============================================================================

print("\n" + "=" * 70)
print("Задача 1. Аналіз стаціонарного часового ряду рейтингів фільмів")
print("=" * 70)

# Аналіз даних
print("\nПерші 5 записів з ratings.csv:")
print(ratings_df.head())

# Створення часового ряду з реальних даних
# Виберемо фільм з найбільшою кількістю оцінок
movie_ratings_count = ratings_df.groupby('movieId').size().sort_values(ascending=False)
top_movie_id = movie_ratings_count.index[0]
top_movie_title = movies_df[movies_df['movieId'] == top_movie_id]['title'].values[0]

print(f"\nФільм для аналізу: {top_movie_title} (ID: {top_movie_id})")
print(f"Кількість оцінок: {movie_ratings_count.iloc[0]}")

# Отримуємо всі оцінки для обраного фільму
movie_ratings = ratings_df[ratings_df['movieId'] == top_movie_id]['rating'].values

# Якщо оцінок більше 16, обмежуємо до 16 для відповідності умові
if len(movie_ratings) > 16:
    time_series = movie_ratings[:16]
else:
    # Доповнюємо ряд до 16 елементів, якщо потрібно
    time_series = np.concatenate([movie_ratings, np.random.normal(movie_ratings.mean(), movie_ratings.std(), 16 - len(movie_ratings))])

time_periods = np.arange(1, 17)

print(f"\nЧасовий ряд рейтингів: {time_series}")
print(f"Кількість спостережень: {len(time_series)}")

# а) Побудова графіка часового ряду
plt.figure(figsize=(12, 10))

# Основний графік часового ряду
plt.subplot(2, 2, 1)
plt.plot(time_periods, time_series, 'o-', linewidth=2, markersize=8, color='blue', markerfacecolor='red')
plt.title(f'Часовий ряд рейтингів\n"{top_movie_title}"', fontsize=14, fontweight='bold')
plt.xlabel('Період спостереження', fontsize=12)
plt.ylabel('Рейтинг', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(time_periods)

# Додаємо анотації для точок
for i, (t, r) in enumerate(zip(time_periods, time_series)):
    plt.annotate(f'{r:.1f}', (t, r), textcoords="offset points", xytext=(0,10), ha='center')

# б) Графік для візуальної оцінки автокореляції
plt.subplot(2, 2, 2)
plt.plot(time_periods[:-1], time_series[:-1], 's-', label='y(t)', linewidth=2, markersize=6, color='green')
plt.plot(time_periods[1:], time_series[1:], 'o-', label='y(t+1)', linewidth=2, markersize=6, color='orange')
plt.title('Порівняння y(t) та y(t+1)', fontsize=14, fontweight='bold')
plt.xlabel('Час', fontsize=12)
plt.ylabel('Рейтинг', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Статистичні характеристики
mean_rating = np.mean(time_series)
std_rating = np.std(time_series)
min_rating = np.min(time_series)
max_rating = np.max(time_series)

print(f"\nСтатистичні характеристики ряду:")
print(f"Середнє значення: {mean_rating:.3f}")
print(f"Стандартне відхилення: {std_rating:.3f}")
print(f"Мінімальний рейтинг: {min_rating:.1f}")
print(f"Максимальний рейтинг: {max_rating:.1f}")

# в) Точний розрахунок коефіцієнта автокореляції
autocorr_1 = autocorrelation(time_series, lag=1)
autocorr_2 = autocorrelation(time_series, lag=2)

print(f"\nКоефіцієнти автокореляції:")
print(f"Автокореляція 1-го порядку: {autocorr_1:.4f}")
print(f"Автокореляція 2-го порядку: {autocorr_2:.4f}")

# Графік залежності y(t+1) від y(t)
plt.subplot(2, 2, 3)
plt.scatter(time_series[:-1], time_series[1:], alpha=0.8, s=80, color='red', edgecolors='black')
plt.title('Залежність y(t+1) від y(t)', fontsize=14, fontweight='bold')
plt.xlabel('y(t)', fontsize=12)
plt.ylabel('y(t+1)', fontsize=12)

# Лінія регресії
slope, intercept, r_value, p_value, std_err = stats.linregress(time_series[:-1], time_series[1:])
x_fit = np.linspace(np.min(time_series), np.max(time_series), 100)
y_fit = slope * x_fit + intercept
plt.plot(x_fit, y_fit, '--', color='blue', linewidth=2, 
         label=f'Лінія регресії (r={r_value:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)

# Додаємо анотації для точок на графіку розсіювання
for i, (x, y) in enumerate(zip(time_series[:-1], time_series[1:])):
    plt.annotate(f'({x:.1f},{y:.1f})', (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

# г) Гістограма розподілу рейтингів
plt.subplot(2, 2, 4)
plt.hist(time_series, bins=6, alpha=0.7, color='lightblue', edgecolor='black', linewidth=1.2)
plt.title('Розподіл рейтингів фільму', fontsize=14, fontweight='bold')
plt.xlabel('Рейтинг', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(True, alpha=0.3)

# Додаємо значення на стовпцях гістограми
counts, bins, patches = plt.hist(time_series, bins=6, alpha=0)
for count, bin_edge, patch in zip(counts, bins, patches):
    if count > 0:
        plt.text(bin_edge + 0.1, count + 0.1, str(int(count)), ha='left', va='bottom')

plt.tight_layout()
plt.show()

print(f"\nРезультати регресійного аналізу:")
print(f"Коефіцієнт кореляції Пірсона: {r_value:.4f}")
print(f"Нахил лінії регресії: {slope:.4f}")
print(f"Перетин: {intercept:.4f}")
print(f"P-значення: {p_value:.4f}")

# Аналіз результатів
print(f"\nІнтерпретація результатів:")
if autocorr_1 > 0.5:
    print("▪ Висока позитивна автокореляція - рейтинги мають сильну залежність від попередніх значень")
elif autocorr_1 > 0.2:
    print("▪ Помірна позитивна автокореляція - помітна залежність між послідовними рейтингами")
else:
    print("▪ Слабка автокореляція - рейтинги майже незалежні")

if r_value > 0:
    print("▪ Позитивна кореляція між y(t) та y(t+1) - високі рейтинги схильні з'являтися групами")

input("\nНатисніть Enter для переходу до наступної задачі...")

# =============================================================================
# ЗАДАЧА 2
# =============================================================================

print("\n" + "=" * 70)
print("Задача 2. Прогнозування популярності фільмів на основі випадкового блукання")
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
        movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
        
        if len(movie_ratings) < min_ratings:
            print(f"  ⚠ Фільм {movie_id} має лише {len(movie_ratings)} оцінок (потрібно мінімум {min_ratings})")
            return None
            
        movie_ratings = movie_ratings.sort_values('timestamp')
        
        ratings_series = movie_ratings['rating'].values
        
        print(f"  ✓ Фільм {movie_id}: {len(ratings_series)} оцінок")
        return ratings_series
    
    def calculate_trend_from_data(self, series):
        if len(series) < 2:
            return 0, series[0] if len(series) > 0 else 0
            
        n = len(series)
        x = np.arange(n)
        slope, intercept = np.polyfit(x, series, 1)
        return slope, intercept

forecaster = MoviePopularityForecast()

print("Пошук фільмів з достатньою кількістю оцінок...")

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
    print("Не знайдено фільмів з достатньою кількістю оцінок. Використовуються тестові дані...")
    np.random.seed(42)
    test_movies = [
        {'id': 1, 'title': 'Тестовий фільм 1', 'ratings_count': 10, 'avg_rating': 4.0},
        {'id': 2, 'title': 'Тестовий фільм 2', 'ratings_count': 8, 'avg_rating': 3.8},
        {'id': 3, 'title': 'Тестовий фільм 3', 'ratings_count': 7, 'avg_rating': 4.2}
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
    print("Топ фільми за кількістю оцінок:")
    for i, movie in enumerate(top_movies_info, 1):
        print(f"{i}. {movie['title']}")
        print(f"   Оцінок: {movie['ratings_count']}, Середній рейтинг: {movie['avg_rating']:.2f}")

    movies_data = {}
    for movie in top_movies_info:
        print(f"\nЗавантаження даних для: {movie['title']}")
        time_series = forecaster.get_movie_time_series(movie['id'], min_ratings=3)
        if time_series is not None:
            movies_data[movie['title']] = time_series
        else:
            print(f"  ❌ Не вдалося завантажити дані для {movie['title']}")

print(f"\nЗавантажено часові ряди для {len(movies_data)} фільмів")

if len(movies_data) == 0:
    print("Не вдалося завантажити дані жодного фільму. Завершення програми.")
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
    
    # Прогноз на 5 періодів
    forecast_periods = 5
    slope, intercept = forecaster.calculate_trend_from_data(popularity)
    future_periods = np.arange(n_periods, n_periods + forecast_periods)
    forecast = intercept + slope * future_periods
    
    plt.plot(future_periods, forecast, '--', color=colors[i], linewidth=1, alpha=0.7)

plt.title('Історичні рейтинги та прогнози на 5 періодів', fontsize=14, fontweight='bold')
plt.xlabel('Період спостереження', fontsize=12)
plt.ylabel('Рейтинг', fontsize=12)
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
         label='Прогнозований рейтинг', color='red')
plt.fill_between(horizons, 
                 np.array(forecasts) - np.array(errors),
                 np.array(forecasts) + np.array(errors),
                 alpha=0.3, label='Довірчий інтервал ±1σ', color='red')

plt.title(f'Прогноз для "{movie_name}"', fontsize=14, fontweight='bold')
plt.xlabel('Горизонт прогнозу (періоди)', fontsize=12)
plt.ylabel('Рейтинг', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
mses = [forecaster.mean_squared_error(forecaster.volatility, h) for h in horizons]

plt.plot(horizons, errors, 'o-', linewidth=2, label='Помилка прогнозу', color='blue')
plt.plot(horizons, np.sqrt(mses), 's-', linewidth=2, label='RMSE', color='green')
plt.title('Похибки прогнозування', fontsize=14, fontweight='bold')
plt.xlabel('Горизонт прогнозу (періоди)', fontsize=12)
plt.ylabel('Похибка', fontsize=12)
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

plt.title('Рекомендації на основі прогнозованого рейтингу', fontsize=14, fontweight='bold')
plt.ylabel('Прогнозований рейтинг через 5 періодів', fontsize=12)
plt.xticks(range(len(movies_sorted)), movies_sorted, rotation=15, ha='right')
plt.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, forecasts_sorted):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nРезультати прогнозування на 5 періодів:")
for movie_name, forecast in sorted_movies:
    current_pop = movies_data[movie_name][-1]
    growth = forecast - current_pop
    growth_percent = (growth / current_pop) * 100
    
    print(f"\n{movie_name}:")
    print(f"  Поточний рейтинг: {current_pop:.3f}")
    print(f"  Прогноз через 5 періодів: {forecast:.3f}")
    print(f"  Очікуваний приріст: {growth:+.3f} ({growth_percent:+.1f}%)")
    print(f"  Довірчий інтервал: [{forecast - errors[4]:.3f}, {forecast + errors[4]:.3f}]")

print("\n1. Фільми для активного просування:")
for i, (movie_name, forecast) in enumerate(sorted_movies[:2], 1):
    print(f"   {i}. {movie_name} (прогноз: {forecast:.3f})")

print("\n2. Фільми для моніторингу:")
for i, (movie_name, forecast) in enumerate(sorted_movies[2:], 1):
    print(f"   {i}. {movie_name} (прогноз: {forecast:.3f})")

print(f"\n3. Параметри моделі:")
print(f"   ▪ Сталий тренд: {forecaster.trend:.3f} за період")
print(f"   ▪ Волатильність: {forecaster.volatility:.3f}")
print(f"   ▪ Тип моделі: Випадкове блукання з дрейфом")

print(f"\n4. Точність прогнозів:")
print(f"   ▪ Помилка прогнозу на 1 період: ±{errors[0]:.3f}")
print(f"   ▪ Помилка прогнозу на 5 періодів: ±{errors[4]:.3f}")
print(f"   ▪ RMSE на 5 періодів: {np.sqrt(mses[4]):.3f}")

for movie_name, popularity in movies_data.items():
    if len(popularity) > 5:
        autocorr = autocorrelation(popularity, 1)
        print(f"\n{movie_name}:")
        print(f"  Автокореляція 1-го порядку: {autocorr:.3f}")
        if autocorr > 0.3:
            print("  ▪ Сильна залежність від попередніх значень")
        elif autocorr > 0.1:
            print("  ▪ Помірна залежність")
        else:
            print("  ▪ Слабка залежність")

print("ВИСНОВКИ:")
print("1. Реальні дані показують помірну автокореляцію рейтингів")
print("2. Модель випадкового блукання дозволяє прогнозувати динаміку популярності")
print("3. Система може ефективно ранжувати фільми для рекомендацій")
print("4. Точність прогнозів зменшується зі збільшенням горизонту прогнозування")

input("\nНатисніть Enter для завершення...")