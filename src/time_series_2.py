# DESCRIPTION:
# Оцінити побудовану регресійну модель.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats
import matplotlib.pyplot as plt


# Дані
x = np.array([5, 7, 3, 8, 4, 6, 9, 2, 10, 1]).reshape(-1, 1)  # reshape для моделі
y = np.array([20, 25, 15, 28, 18, 22, 30, 12, 32, 10])


# Побудова моделі
model = LinearRegression()
model.fit(x, y)


# Прогноз
y_pred = model.predict(x)


# Оцінка моделі
r2 = r2_score(y, y_pred)  # R² — коефіцієнт детермінації
rmse = np.sqrt(mean_squared_error(y, y_pred))  # RMSE — корінь середньоквадратичної помилки
mae = mean_absolute_error(y, y_pred)  # MAE — середня абсолютна помилка


# p-value для b1 (t-test)
slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)


print(f"Оцінка моделі:")
print(f"R²: {r2:.4f} (частка поясненої варіації)")
print(f"RMSE: {rmse:.2f} (середньоквадратична помилка)")
print(f"MAE: {mae:.2f} (середня абсолютна помилка)")
print(f"p-value: {p_value:.4f} (значущість, <0.05 — значуща)")
# Приклад: R²: 0.9976, RMSE: 0.44, MAE: 0.35, p-value: 0.0000


# Візуалізація залишків
residuals = y - y_pred
plt.figure(figsize=(10, 5))


plt.subplot(1, 2, 1)  # Графік залишків
plt.scatter(y_pred, residuals, color='green', label='Залишки')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)  # Нульова межа
plt.xlabel('Прогнозовані значення')
plt.ylabel('Залишки')
plt.title('Аналіз залишків')
plt.legend()
plt.grid(True)


# Показати графіки і чекати закриття
plt.tight_layout()
plt.show(block=True)
