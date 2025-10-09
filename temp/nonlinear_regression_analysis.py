# DESCRIPTION:
# Виконати нелінійний регресійний аналіз із використанням Fuzzy Logic Toolbox.
import numpy as np
import statsmodels.api as sm


# Дані з таблиці 2.6
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y_data = np.array([3-1, 3-1, 3-1, 2-1, 2-1, 2-1, 5-1, 5-1, 4-1, 4-1, 4-1])


# Додавання константи для моделі
X = sm.add_constant(np.vstack([x_data, x_data**2]).T)
model = sm.OLS(y_data, X).fit()


# Виведення результатів
print(model.summary())
a, b, c = model.params
print(f'Модель: y = {a:.4f} + {b:.4f}x + {c:.4f}x^2')


# Прогнозування
y_pred = model.predict(X)
print('Прогнозовані значення:', y_pred)
