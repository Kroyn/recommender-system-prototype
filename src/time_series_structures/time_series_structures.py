# --------------------------------------------------------------------------------
# TODO: Organize the fragments, make a selection, and possibly even translate it.
# --------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns

plt.ion() 

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

merged = ratings.merge(movies, on='movieId')

yearly_stats = merged.groupby('year').agg({
    'rating': ['mean', 'count']
}).round(3)
yearly_stats.columns = ['avg_rating', 'rating_count']
yearly_stats = yearly_stats.dropna()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(yearly_stats.index, yearly_stats['avg_rating'], marker='o')
plt.title('Середній рейтинг фільмів по роках')
plt.xlabel('Рік')
plt.ylabel('Рейтинг')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(yearly_stats.index, yearly_stats['rating_count'], marker='s', color='red')
plt.title('Кількість оцінок по роках')
plt.xlabel('Рік')
plt.ylabel('Кількість оцінок')
plt.grid(True)

plt.tight_layout()
plt.show(block=False)
plt.pause(2) 

X = yearly_stats.index.values.reshape(-1, 1)
y = yearly_stats['avg_rating'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)

print(f'Коефіцієнт детермінації R²: {r2:.4f}')
print(f'Рівняння: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x')

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred, color='red', label='Лінія регресії')
plt.title('Лінійна регресія: залежність рейтингу від року')
plt.xlabel('Рік')
plt.ylabel('Середній рейтинг')
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(2)

future_years = np.array([2024, 2025]).reshape(-1, 1)
future_ratings = model.predict(future_years)

print(f'Прогноз середнього рейтингу на 2024 рік: {future_ratings[0]:.3f}')
print(f'Прогноз середнього рейтингу на 2025 рік: {future_ratings[1]:.3f}')

X_count = yearly_stats.index.values.reshape(-1, 1)
y_count = yearly_stats['rating_count'].values

model_count = LinearRegression()
model_count.fit(X_count, y_count)
future_counts = model_count.predict(future_years)

print(f'Прогноз кількості оцінок на 2024: {future_counts[0]:.0f}')
print(f'Прогноз кількості оцінок на 2025: {future_counts[1]:.0f}')

corr = yearly_stats.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Матриця кореляцій')
plt.show(block=False)
plt.pause(2)

yearly_stats['rating_ma'] = yearly_stats['avg_rating'].rolling(window=3, center=True).mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_stats.index, yearly_stats['avg_rating'], label='Оригінал', alpha=0.7)
plt.plot(yearly_stats.index, yearly_stats['rating_ma'], label='Згладжено (MA)', linewidth=2)
plt.title('Згладжування часового ряду середнього рейтингу')
plt.xlabel('Рік')
plt.ylabel('Рейтинг')
plt.legend()
plt.grid(True)
plt.show(block=False)
plt.pause(2)

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
plt.title('Середній рейтинг за віком користувачів (17-50)')
plt.xlabel('Вік')
plt.ylabel('Середній рейтинг')
plt.xticks(rotation=45)
plt.grid(True)
plt.show(block=False)
plt.pause(2)

top_movies = target_ratings.groupby('movieId')['rating'].mean().nlargest(10)
top_movies_with_titles = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title']]
top_movies_with_titles = top_movies_with_titles.merge(top_movies, on='movieId')
print("Топ-10 фільмів для аудиторії 17-50:")
print(top_movies_with_titles.round(3))

input("Натисніть Enter для завершення...")
plt.close('all')  
