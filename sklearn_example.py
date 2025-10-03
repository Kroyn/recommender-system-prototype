# DESCRIPTION:
# Приклад реалізації моделі машинного навчання та її візуалізації на Python.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay

# Генеруємо синтетичні дані для прикладу
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Створюємо та навчаємо модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Візуалізуємо ROC-криву
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Випадкове вгадування', alpha=0.8)
ax.set_title('ROC-крива для моделі Випадкового Лісу')
ax.legend(loc="lower right")
plt.grid(True)
plt.show()
