# DESCRIPTION:
# Класифікація видів ірисів за ознаками (довжина/ширина пелюсток/чашолистків) за допомогою LDA.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. Дані
iris = load_iris()
X = iris.data
y = iris.target


# 2. Розділ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 3. Модель LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)


# 4. Прогноз
y_pred = lda.predict(X_test)


# 5. Оцінка
print(f"Точність: {accuracy_score(y_test, y_pred)}")


# 6. Візуалізація (проєкція на 2D)
X_lda = lda.transform(X)
plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='viridis')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA проекція ірисів')
plt.show()
