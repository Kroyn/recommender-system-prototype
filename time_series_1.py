# DESCRIPTION:
# Визначити та побудувати функцію регресії, вплив змінних.
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Дані
x = np.array([5, 7, 3, 8, 4, 6, 9, 2, 10, 1]).reshape(-1, 1)  # reshape для моделі
y = np.array([20, 25, 15, 28, 18, 22, 30, 12, 32, 10])


# Побудова моделі
model = LinearRegression()
model.fit(x, y)


# Коефіцієнти
b0 = model.intercept_
b1 = model.coef_[0]
print(f"Рівняння: Y = {b0:.2f} + {b1:.2f} * X")
# Вплив: Зростання X на 1 тис. грн збільшує Y на 2.20 тис. одиниць.


# Прогноз
y_pred = model.predict(x)


# Побудова графіка
plt.scatter(x, y, color='blue', label='Дані')  # Точки даних
plt.plot(x, y_pred, color='red', linestyle='--', label='Лінія регресії')  # Лінія регресії
plt.xlabel('Витрати на рекламу (тис. грн)')
plt.ylabel('Продажі (тис. одиниць)')
plt.title('Лінійна регресія')
plt.legend()
plt.grid(True)


# Показати графік і чекати закриття користувачем
plt.show(block=True)
