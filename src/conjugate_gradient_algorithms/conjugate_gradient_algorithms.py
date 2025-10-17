import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

monthly_ratings = ratings.groupby('month')['rating'].mean()

plt.figure(figsize=(14, 8))
plt.plot(monthly_ratings.index.astype(str), monthly_ratings.values, 'o-', linewidth=2, markersize=6)
plt.title('Динаміка середнього рейтингу по місяцях', fontsize=14, fontweight='bold')
plt.xlabel('Місяць', fontsize=12)
plt.ylabel('Середній рейтинг', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(3.5, 5.0)

x_numeric = np.arange(len(monthly_ratings))
z = np.polyfit(x_numeric, monthly_ratings.values, 1)
p = np.poly1d(z)
plt.plot(monthly_ratings.index.astype(str), p(x_numeric), "r--", alpha=0.8, label='Тренд')

plt.legend()
plt.tight_layout()
plt.show()

print(f"Період аналізу: від {monthly_ratings.index.min()} до {monthly_ratings.index.max()}")
print(f"Кількість місяців: {len(monthly_ratings)}")
print(f"Середній рейтинг за весь період: {monthly_ratings.mean():.3f}")

daily_ratings = ratings.groupby('date')['rating'].mean().resample('D').mean()
daily_ratings = daily_ratings.fillna(method='ffill')

decomposition = seasonal_decompose(daily_ratings.head(365), model='additive', period=30)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.suptitle('Декомпозиція часового ряду рейтингів фільмів')
plt.tight_layout()
plt.show()

print("Компоненти ряду:")
print("- Тренд: Загальна тенденція зміни рейтингів")
print("- Сезонність: Щомісячні/щотижневі коливання")
print("- Залишкова складова: Випадкові коливання")

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

merged = ratings.merge(movies, on='movieId')

yearly_avg = merged.groupby('year')['rating'].mean().dropna()
years = yearly_avg.index.values.reshape(-1, 1)
ratings_avg = yearly_avg.values

model = LinearRegression()
model.fit(years, ratings_avg)
trend_line = model.predict(years)

plt.figure(figsize=(12, 6))
plt.plot(yearly_avg.index, yearly_avg.values, 'o-', label='Середній рейтинг', alpha=0.7)
plt.plot(yearly_avg.index, trend_line, 'r--', linewidth=2, label='Лінія тренду')
plt.title('Тренд середнього рейтингу фільмів по роках')
plt.xlabel('Рік випуску')
plt.ylabel('Середній рейтинг')
plt.legend()
plt.grid(True)
plt.show()

print(f"Рівняння тренду: y = {model.intercept_:.4f} + {model.coef_[0]:.4f} * x")
print(f"Коефіцієнт детермінації R²: {model.score(years, ratings_avg):.4f}")

monthly_data = ratings.groupby('month')['rating'].mean()

ma_3 = monthly_data.rolling(window=3, center=True).mean()
ma_6 = monthly_data.rolling(window=6, center=True).mean()

exp_smooth = monthly_data.ewm(span=3, adjust=False).mean()

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(monthly_data.index.astype(str), monthly_data.values, 'o-', label='Оригінал', alpha=0.5)
plt.plot(monthly_data.index.astype(str), ma_3.values, 'r-', label='Ковзне середнє (3 міс.)')
plt.plot(monthly_data.index.astype(str), ma_6.values, 'g-', label='Ковзне середнє (6 міс.)')
plt.title('Згладжування методом ковзного середнього')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
plt.plot(monthly_data.index.astype(str), monthly_data.values, 'o-', label='Оригінал', alpha=0.5)
plt.plot(monthly_data.index.astype(str), exp_smooth.values, 'purple', label='Експоненціальне згладжування')
plt.title('Експоненціальне згладжування')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("Порівняння методів згладжування:")
print("- Ковзне середнє: Простий метод, добре видаляє шум")
print("- Експоненціальне: Більша вага останнім спостереженням")

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
plt.xlabel('Середній рейтинг')
plt.ylabel('Кількість оцінок')
plt.title('Кластерний аналіз фільмів (аналог STATISTICA)')
plt.grid(True)
plt.show()

print("Застосування модулів STATISTICA в проекті:")
print("- Time Series Analysis: Аналіз динаміки рейтингів")
print("- Data Miner: Побудова рекомендаційних моделей")
print("- Neural Networks: Глибоке навчання для рекомендацій")