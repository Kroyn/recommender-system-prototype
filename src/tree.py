import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

movies = pd.read_csv('database/movies.csv')
ratings = pd.read_csv('database/ratings.csv')
users = pd.read_csv('database/users.csv')

data = ratings.merge(movies, on='movieId').merge(users, on='userId')

le_genre = LabelEncoder()
data['genre_encoded'] = le_genre.fit_transform(data['genres'].str.split('|').str[0])

data['high_rating'] = (data['rating'] > 4.0).astype(int)

features = ['genre_encoded', 'age', 'gender']
X = pd.get_dummies(data[features], columns=['gender'])
y = data['high_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.4f}")

plt.figure(figsize=(20,12))
plot_tree(model, 
          feature_names=X.columns,
          class_names=['Low Rating', 'High Rating'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Дерево рішень для прогнозування високих рейтингів')
plt.show()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Важливість ознак:")
print(feature_importance)
