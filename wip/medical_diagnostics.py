# DESCRIPTION:
# Медична діагностика
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score


# Завантажуємо дані
iris = datasets.load_iris()
X = iris.data[:, :2]  # Беремо тільки дві перші ознаки для простоти візуалізації
y = (iris.target != 0).astype(int)  # Перетворюємо задачу на бінарну: клас 0 vs решта


# Розділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Створюємо та навчаємо модель логістичної регресії
model = LogisticRegression()
model.fit(X_train, y_train)


# Робимо передбачення на тестовій вибірці
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1] # ймовірності для класу 1


# Оцінюємо точність моделі
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}")


# Візуалізуємо межі прийняття рішення (для 2D)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette='viridis')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
plt.colorbar()
plt.title("Логістична регресія: Межі класифікації")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
