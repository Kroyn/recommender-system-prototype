# DESCRIPTION:
# Основні стадії Data Mining з прикладами з проєкту.
import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([5, 7, 3, 8, 4, 6, 9, 2, 10, 1])
y = np.array([20, 25, 15, 28, 18, 22, 30, 12, 32, 10])

# Calculating correlation
correlation = np.corrcoef(x, y)[0, 1]
print(f"Коефіцієнт кореляції Пірсона: {correlation}")

# Graph construction
plt.scatter(x, y, color='blue', label='Дані')  # Scatter plot
plt.xlabel('Витрати на рекламу (тис. грн)')
plt.ylabel('Продажі (тис. одиниць)')
plt.title('Зв’язок між витратами на рекламу та продажами')


# Adding a trend line
z = np.polyfit(x, y, 1)  # Linear regression (1 — degree of polynomial)
p = np.poly1d(z)  # Creating a trend line with a log
plt.plot(x, p(x), color='red', linestyle='--', label='Лінія тренду')


plt.legend()
plt.grid(True) # Add grid if don't know
plt.show()
