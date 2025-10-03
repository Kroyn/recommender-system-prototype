# DESCRIPTION:
# Точність, AUC-ROC, матриця помилок.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)


# 1. ЗАВАНТАЖУЄМО ТА ГОТУЄМО ДАНІ
iris = load_iris()
X = iris.data[:, :2]  # Використовуємо лише дві перші ознаки
y = (iris.target != 0).astype(int)  # Перетворюємо на бінарну задачу


# 2. РОЗДІЛЯЄМО ДАНІ НА НАВЧАЛЬНУ ТА ТЕСТОВУ ВИБІРКИ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)


# 3. СТВОРЮЄМО ТА НАВЧАЄМО МОДЕЛЬ
model = LogisticRegression()
model.fit(X_train, y_train)


# 4. РОБИМО ПЕРЕДБАЧЕННЯ
y_pred = model.predict(X_test)  # Передбачені класи (0 або 1)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Ймовірність класу 1


# 5. ОЦІНЮЄМО МОДЕЛЬ


# 5.1. Точність
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}\n")


# 5.2. Матриця помилок
cm = confusion_matrix(y_test, y_pred)
print("Матриця помилок:")
print(cm)
print("\n(Рядки: Факт, Стовпці: Прогноз)")
print("[[TN FP]\n [FN TP]]")


# Візуалізація матриці помилок
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Клас 0', 'Клас 1'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Матриця помилок')
plt.show()


# 5.3. ROC-крива та AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)


print(f"Площа під ROC-кривою (AUC): {roc_auc:.3f}")


# Візуалізація ROC-кривої
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Випадкове вгадування (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-крива')
plt.legend(loc="lower right")
plt.show()


# 6. ДЕМОНСТРАЦІЯ РОБОТИ МОДЕЛІ НА ПРИКЛАДІ
print("\nДемонстрація прогнозу для нових даних:")
new_data = np.array([[5.0, 3.5],  # Приклад 1
                     [6.5, 2.8]]) # Приклад 2


new_pred = model.predict(new_data)
new_prob = model.predict_proba(new_data)


for i in range(len(new_data)):
    print(f"Об'єкт {i+1}: {new_data[i]}")
    print(f"  Прогноз: {new_pred[i]} ('Клас 1'? {bool(new_pred[i])})")
    print(f"  Впевненість: [Йм-ть Класу 0: {new_prob[i][0]:.3f}, "
          f"Йм-ть Класу 1: {new_prob[i][1]:.3f}]")
    print()

