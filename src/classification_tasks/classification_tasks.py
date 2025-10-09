# --------------------------------------------------------------------------------
# TODO: Organize the fragments, make a selection, and possibly even translate it.
# --------------------------------------------------------------------------------
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))

movies_path = os.path.join(script_dir, 'movies.csv')
ratings_path = os.path.join(script_dir, 'ratings.csv')

try:
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    print("Дані успішно завантажено")
    print(f"Фільмів: {len(movies)}, Рейтингів: {len(ratings)}")
except Exception as e:
    print(f"Помилка завантаження даних: {e}")
    exit()

movie_stats = ratings.groupby('movieId').agg({
    'rating': ['mean', 'count', 'std'],
    'userId': 'nunique'
}).round(3)
movie_stats.columns = ['avg_rating', 'rating_count', 'rating_std', 'unique_users']

movie_stats['popularity_score'] = (
    movie_stats['avg_rating'] *
    movie_stats['rating_count'] *
    (1 - movie_stats['rating_std']/5)
)

threshold_upper = movie_stats['popularity_score'].quantile(0.75)
threshold_lower = movie_stats['popularity_score'].quantile(0.25)

movie_stats['popularity_class'] = pd.cut(
    movie_stats['popularity_score'],
    bins=[-float('inf'), threshold_lower, threshold_upper, float('inf')],
    labels=['low', 'medium', 'high']
)

print("\nРозподіл фільмів за популярністю:")
print(movie_stats['popularity_class'].value_counts())

X = movie_stats[['avg_rating', 'rating_count', 'rating_std', 'unique_users']]
y = movie_stats['popularity_class']

def prepare_data(X, y):
    common_index = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[common_index]
    y_clean = y.loc[common_index]
    return X_clean, y_clean

X, y = prepare_data(X, y)
print(f"\nРозмірність даних після очищеня: X{X.shape}, y{y.shape}")
print("\nДетальний розподіл класів:")
class_distribution = y.value_counts()
print(class_distribution)

def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Викиди в {column}: {len(outliers)} записів")

    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

for column in ['avg_rating', 'rating_count']:
    movie_stats = handle_outliers_iqr(movie_stats, column)

def data_quality_report(X, y):
    print("\nЗВІТ ПРО ЯКІСТЬ ДАНИХ")
    print(f"Розмір X: {X.shape}")
    print(f"Розмір y: {y.shape}")
    print(f"Пропуски в X: {X.isna().sum().sum()}")
    print(f"Пропуски в y: {y.isna().sum()}")
    print(f"Баланс класів:")
    print(y.value_counts(normalize=True).round(3))
    print(f"Мінімальна кількість елементів у класі: {y.value_counts().min()}")
   
data_quality_report(X, y)

mi_scores = mutual_info_classif(X, y, random_state=42)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': mi_scores
}).sort_values('importance', ascending=False)

print("\nІнформативність ознак:")
print(feature_importance)

def binary_classification_example():
    movie_stats['is_highly_rated'] = (movie_stats['avg_rating'] >= 4.0).astype(int)
    
    X_binary = movie_stats[['rating_count', 'unique_users', 'rating_std']]
    y_binary = movie_stats['is_highly_rated']

    X_binary = X_binary.dropna()
    y_binary = y_binary[X_binary.index]
    
    model = LogisticRegression(random_state=42)
    model.fit(X_binary, y_binary)
    
    accuracy = model.score(X_binary, y_binary)
    print(f"Точність бінарної класифікації: {accuracy:.3f}")
    
    return model

def multiclass_classification_example():
    X_multi = movie_stats[['rating_count', 'unique_users', 'rating_std', 'avg_rating']]
    y_multi = movie_stats['popularity_class']
    
    X_multi = X_multi.dropna()
    y_multi = y_multi[X_multi.index]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_multi, y_multi)
    
    accuracy = model.score(X_multi, y_multi)
    print(f"Точність мультикласової класифікації: {accuracy:.3f}")
    
    return model

print("=== БІНАРНА КЛАСИФІКАЦІЯ ===")
binary_model = binary_classification_example()

print("\n=== МУЛЬТИКЛАСОВА КЛАСИФІКАЦІЯ ===")
multiclass_model = multiclass_classification_example()

def multilabel_genre_classification():
    np.random.seed(42)
    n_movies = len(movie_stats)
    genres = ['Action', 'Comedy', 'Drama', 'Romance', 'Thriller']

    genre_labels = pd.DataFrame(
        np.random.randint(0, 2, size=(n_movies, len(genres))),
        columns=genres,
        index=movie_stats.index
    )

    X_ml = movie_stats[['avg_rating', 'rating_count', 'rating_std']].dropna()
    y_ml = genre_labels.loc[X_ml.index]
    
    print(f"\nРозмірність даних для мультилейбл класифікації: {X_ml.shape}")
    print("Розподіл жанрів:")
    print(y_ml.sum().sort_values(ascending=False))
    
    return X_ml, y_ml

X_ml, y_ml = multilabel_genre_classification()

def compare_classification_algorithms(X, y):
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    results = {}
 
    min_class_size = y.value_counts().min()
    n_splits = min(5, min_class_size)
   
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        print("Використовується train-test split через малу кількість даних")
   
    for name, model in algorithms.items():
        start_time = time.time()
       
        try:
            if n_splits >= 2:
                if name in ['SVM', 'K-NN', 'Logistic Regression', 'Naive Bayes']:
                    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
                else:
                    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
               
                mean_accuracy = scores.mean()
                std_accuracy = scores.std()
                method = f"CV ({n_splits}-fold)"
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled if name in ['SVM', 'K-NN', 'Logistic Regression', 'Naive Bayes'] else X,
                    y, test_size=0.3, random_state=42, stratify=y
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mean_accuracy = accuracy_score(y_test, y_pred)
                std_accuracy = 0
                method = "train-test split"
           
            execution_time = time.time() - start_time
            results[name] = {
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'time_seconds': execution_time,
                'method': method
            }
           
            print(f"{name:20} | Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f} | Time: {execution_time:.2f}s | {method}")
           
        except Exception as e:
            print(f"{name:20} | ERROR: {e}")
            results[name] = {
                'mean_accuracy': 0,
                'std_accuracy': 0,
                'time_seconds': 0,
                'method': 'failed'
            }
   
    return results

print("\n" + "="*70)
print("ПОРІВНЯННЯ АЛГОРИТМІВ КЛАСИФІКАЦІЇ")
print("="*70)

X_compare = movie_stats[['avg_rating', 'rating_count', 'rating_std', 'unique_users']]
y_compare = movie_stats['popularity_class']

X_compare, y_compare = prepare_data(X_compare, y_compare)

results = compare_classification_algorithms(X_compare, y_compare)

if results and any(r['mean_accuracy'] > 0 for r in results.values()):
    results_df = pd.DataFrame(results).T
    results_df = results_df[results_df['mean_accuracy'] > 0].sort_values('mean_accuracy', ascending=False)

    if not results_df.empty:
        plt.figure(figsize=(12, 6))

        y_min = max(0, results_df['mean_accuracy'].min() - 0.1)
        y_max = min(1.0, results_df['mean_accuracy'].max() + 0.1)
       
        bars = plt.bar(results_df.index, results_df['mean_accuracy'],
                       yerr=results_df['std_accuracy'], capsize=5, alpha=0.7,
                       color=['#2E8B57', '#4682B4', '#FF6347', '#FFD700', '#9370DB', '#20B2AA'])

        plt.title('Порівняння точності алгоритмів класифікації', fontsize=14, fontweight='bold')
        plt.ylabel('Точність (Accuracy)', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(y_min, y_max)
        plt.grid(axis='y', alpha=0.3)

        for bar, accuracy in zip(bars, results_df['mean_accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

        best_model = results_df.index[0]
        best_accuracy = results_df['mean_accuracy'].iloc[0]
        print(f"\nНайкращий алгоритм: {best_model} з точністю {best_accuracy:.3f}")

print(f"\nЗагальний розмір даних для навчання: {X.shape}")
print("\nАналіз завершено!")

print("\nДодаткова інформація:")
print(f"Усього фільмів з рейтингами: {len(movie_stats)}")
print(f"Фільми без рейтингів: {len(movies) - len(movie_stats)}")
print(f"Відсоток охоплення: {len(movie_stats)/len(movies)*100:.1f}%")

def detailed_rf_analysis(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    n_samples = len(X)
    if n_samples < 50:
        train_sizes = np.linspace(0.3, 1.0, 6)
    else:
        train_sizes = np.linspace(0.1, 1.0, 10)

    cv_folds = min(5, n_samples // 2, len(np.unique(y)))
    if cv_folds < 2:
        cv_folds = 2

    train_sizes, train_scores, test_scores = learning_curve(
        rf, X, y, cv=cv_folds, scoring='accuracy', 
        train_sizes=train_sizes,
        random_state=42
    )
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Навчальна вибірка')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Тестова вибірка')
    plt.xlabel('Розмір навчальної вибірки')
    plt.ylabel('Точність')
    plt.title('Крива навчання Random Forest')
    plt.legend()
    plt.grid(True)

    rf.fit(X, y)
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.subplot(1, 2, 2)
    plt.barh(feature_imp['feature'], feature_imp['importance'])
    plt.xlabel('Важливість ознаки')
    plt.title('Важливість ознак у Random Forest')
    plt.tight_layout()
    plt.show()
    
    return rf, feature_imp

rf_model, feature_importance = detailed_rf_analysis(X_compare, y_compare)
print("Важливість ознак у моделі:")
print(feature_importance)

def comprehensive_comparison(X, y_true=None, max_clusters=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering_algorithms = {
        'K-Means': KMeans(n_clusters=3, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=3)
    }
    
    results = {}

    for name, algorithm in clustering_algorithms.items():
        print(f"\n--- {name} ---")

        if name == 'DBSCAN':
            labels = algorithm.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"Кількість кластерів: {n_clusters}")
            print(f"Точки шуму: {n_noise}")
        else:
            labels = algorithm.fit_predict(X_scaled)
            n_clusters = len(set(labels))
            print(f"Кількість кластерів: {n_clusters}")

        if n_clusters > 1:
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            davies = davies_bouldin_score(X_scaled, labels)
            
            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Calinski-Harabasz: {calinski:.3f}")
            print(f"Davies-Bouldin: {davies:.3f}")
            
            results[name] = {
                'labels': labels,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies': davies,
                'n_clusters': n_clusters
            }

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    if y_true is not None:
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('Справжні класи (Classification)')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[0, 0])

    for idx, (name, result) in enumerate(results.items()):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        
        scatter = axes[row, col].scatter(X_pca[:, 0], X_pca[:, 1], c=result['labels'], 
                                        cmap='Set2', alpha=0.7)
        axes[row, col].set_title(f'{name}\nSilhouette: {result["silhouette"]:.3f}')
        axes[row, col].set_xlabel('PC1')
        axes[row, col].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()
    
    return results

print("\nКОМПЛЕКСНЕ ПОРІВНЯННЯ КЛАСИФІКАЦІЇ ТА КЛАСТЕРИЗАЦІЇ")
print("=" * 70)

X_comp = movie_stats[['avg_rating', 'rating_count', 'unique_users']].dropna()
y_comp = movie_stats.loc[X_comp.index, 'popularity_class']

label_map = {'low': 0, 'medium': 1, 'high': 2}
y_comp_numeric = y_comp.map(label_map)

clustering_results = comprehensive_comparison(X_comp, y_comp_numeric)

def practical_applications_comparison():
    applications = {
        'Класифікація': {
            'Задача': 'Прогнозування популярності фільмів',
            'Вхідні дані': 'Історичні рейтинги, метадані фільмів',
            'Результат': 'Мітки "низька", "середня", "висока" популярність',
            'Використання': 'Фільтрація контенту, пріоритезація рекомендацій',
            'Переваги': 'Чіткі інтерпретовані результати, висока точність',
            'Недоліки': 'Вимагає попередньо розмічених даних'
        },
        'Кластеризація': {
            'Задача': 'Сегментація користувачів за поведінкою',
            'Вхідні дані': 'Патерни переглядів, оцінки, частота активності',
            'Результат': 'Групи користувачів зі схожими смаками',
            'Використання': 'Персоналізація рекомендацій, цільовий маркетинг',
            'Переваги': 'Не вимагає міток, виявляє приховані патерни',
            'Недоліки': 'Складність інтерпретації, залежність від параметрів'
        }
    }

    comparison_df = pd.DataFrame(applications).T
    print("\nПОРІВНЯЛЬНА ТАБЛИЦЯ: КЛАСИФІКАЦІЯ vs КЛАСТЕРИЗАЦІЯ")
    print("=" * 80)
    
    for column in comparison_df.columns:
        print(f"\n{column}:")
        for method, value in comparison_df[column].items():
            print(f"  {method}: {value}")
    
    return comparison_df

def recommendation_system_integration():
    print("\n1. КЛАСИФІКАЦІЯ В РЕКОМЕНДАЦІЙНІЙ СИСТЕМІ:")
    print("   - Фільтрація фільмів за популярністю")
    print("   - Визначення 'гітів' для головної сторінки")
    print("   - Категоризація контенту для різних груп користувачів")

    print("\n2. КЛАСТЕРИЗАЦІЯ В РЕКОМЕНДАЦІЙНІЙ СИСТЕМІ:")
    print("   - Сегментація користувачів на 'категорії смаків'")
    print("   - Виявлення нішевих груп для спеціальних рекомендацій")
    print("   - Аналіз поведінкових патернів")

    print("\n3. КОМБІНОВАНИЙ ПІДХІД:")
    print("   - Класифікація для відбору кандидатів")
    print("   - Кластеризація для тонкої настройки рекомендацій")
    print("   - Гібридна система: Content-Based + Collaborative Filtering")

comparison_table = practical_applications_comparison()
recommendation_system_integration()

def user_behavior_clustering_evaluation(X, max_clusters=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_range = range(2, max_clusters + 1)
    metrics = {
        'inertia': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'davies_bouldin': []
    }
    
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            
            metrics['inertia'].append(kmeans.inertia_)

            if len(np.unique(labels)) > 1:
                metrics['silhouette'].append(silhouette_score(X_scaled, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
            else:
                metrics['silhouette'].append(0)
                metrics['calinski_harabasz'].append(0)
                metrics['davies_bouldin'].append(float('inf'))
                
        except Exception as e:
            print(f"Помилка для k={k}: {e}")
            metrics['inertia'].append(float('inf'))
            metrics['silhouette'].append(0)
            metrics['calinski_harabasz'].append(0)
            metrics['davies_bouldin'].append(float('inf'))

    if len(metrics['inertia']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(k_range, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].set_xlabel('Кількість кластерів (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].grid(True, alpha=0.3)

        valid_silhouette = [x for x in metrics['silhouette'] if x > 0]
        if valid_silhouette:
            axes[0, 1].plot(k_range[:len(valid_silhouette)], valid_silhouette, 'go-', linewidth=2, markersize=8)
            axes[0, 1].set_title('Silhouette Score')
            axes[0, 1].set_xlabel('Кількість кластерів (k)')
            axes[0, 1].set_ylabel('Silhouette Score')
            axes[0, 1].grid(True, alpha=0.3)

        valid_calinski = [x for x in metrics['calinski_harabasz'] if x > 0]
        if valid_calinski:
            axes[1, 0].plot(k_range[:len(valid_calinski)], valid_calinski, 'ro-', linewidth=2, markersize=8)
            axes[1, 0].set_title('Calinski-Harabasz Index')
            axes[1, 0].set_xlabel('Кількість кластерів (k)')
            axes[1, 0].set_ylabel('Calinski-Harabasz')
            axes[1, 0].grid(True, alpha=0.3)

        valid_davies = [x for x in metrics['davies_bouldin'] if x < float('inf')]
        if valid_davies:
            axes[1, 1].plot(k_range[:len(valid_davies)], valid_davies, 'mo-', linewidth=2, markersize=8)
            axes[1, 1].set_title('Davies-Bouldin Index')
            axes[1, 1].set_xlabel('Кількість кластерів (k)')
            axes[1, 1].set_ylabel('Davies-Bouldin')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        print("\nОПТИМАЛЬНА КІЛЬКІСТЬ КЛАСТЕРІВ:")
        if valid_silhouette:
            optimal_k_silhouette = k_range[np.argmax(valid_silhouette)]
            print(f"За Silhouette Score: k = {optimal_k_silhouette}")
            
        if valid_calinski:
            optimal_k_calinski = k_range[np.argmax(valid_calinski)]
            print(f"За Calinski-Harabasz: k = {optimal_k_calinski}")
            
        if valid_davies:
            optimal_k_davies = k_range[np.argmin(valid_davies)]
            print(f"За Davies-Bouldin: k = {optimal_k_davies}")

        scores = []
        for k in k_range:
            idx = k - 2
            if (idx < len(metrics['silhouette']) and 
                idx < len(metrics['calinski_harabasz']) and 
                idx < len(metrics['davies_bouldin'])):
                
                silhouette_val = metrics['silhouette'][idx] if metrics['silhouette'][idx] > 0 else 0
                calinski_val = metrics['calinski_harabasz'][idx] if metrics['calinski_harabasz'][idx] > 0 else 0
                davies_val = metrics['davies_bouldin'][idx] if metrics['davies_bouldin'][idx] < float('inf') else 10

                max_calinski = max(metrics['calinski_harabasz']) if max(metrics['calinski_harabasz']) > 0 else 1
                max_davies = max([x for x in metrics['davies_bouldin'] if x < float('inf')]) if any(x < float('inf') for x in metrics['davies_bouldin']) else 1
                
                score = (silhouette_val + 
                        calinski_val / max_calinski -
                        davies_val / max_davies)
                scores.append(score)
        
        if scores and any(score > 0 for score in scores):
            optimal_k_comprehensive = k_range[np.argmax(scores)]
            print(f"Комплексна рекомендація: k = {optimal_k_comprehensive}")
        else:
            optimal_k_comprehensive = 3 
            print(f"Комплексна рекомендація: k = {optimal_k_comprehensive}")
    else:
        optimal_k_comprehensive = 3
        print("Не вдалося обчислити метрики. Використовується k = 3")
    
    return metrics, optimal_k_comprehensive

print("\nКОМПЛЕКСНА ОЦІНКА ЯКОСТІ КЛАСТЕРИЗАЦІЇ КОРИСТУВАЧІВ")
print("=" * 70)

user_behavior = ratings.groupby('userId').agg({
    'rating': ['mean', 'count', 'std'],
    'movieId': 'nunique'
}).round(3)
user_behavior.columns = ['avg_rating', 'ratings_count', 'rating_std', 'unique_movies']
user_behavior = user_behavior.dropna()

print(f"Розмірність даних користувачів: {user_behavior.shape}")
print(f"Перші 5 записів:\n{user_behavior.head()}")

if len(user_behavior) < 3:
    print("Замало даних для кластеризації. Потрібно мінімум 3 користувачі.")
else:
    try:
        metrics, optimal_k = user_behavior_clustering_evaluation(user_behavior, max_clusters=min(8, len(user_behavior)-1))
        print(f"\nОПТИМАЛЬНА КІЛЬКІСТЬ КЛАСТЕРІВ ДЛЯ КОРИСТУВАЧІВ: {optimal_k}")
    except Exception as e:
        print(f"Помилка під час кластеризації користувачів: {e}")

print("\n" + "="*70)
print("\nАНАЛІЗ ПОВНІСТЮ ЗАВЕРШЕНО!")
print("="*70)

def detailed_cluster_analysis(X, k=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    X_clustered = X.copy()
    X_clustered['cluster'] = labels

    cluster_stats = X_clustered.groupby('cluster').agg({
        'avg_rating': ['mean', 'std', 'count'],
        'ratings_count': ['mean', 'std'],
        'unique_movies': ['mean', 'std'],
        'rating_std': ['mean', 'std']
    }).round(3)
    
    print("\nСТАТИСТИКА КЛАСТЕРІВ:")
    print("=" * 50)
    print(cluster_stats)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    cluster_means = X_clustered.groupby('cluster').mean()

    axes[0, 0].bar(cluster_means.index, cluster_means['avg_rating'], 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 0].set_title('Середній рейтинг по кластерам')
    axes[0, 0].set_xlabel('Кластер')
    axes[0, 0].set_ylabel('Середній рейтинг')

    axes[0, 1].bar(cluster_means.index, cluster_means['ratings_count'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0, 1].set_title('Середня кількість оцінок по кластерам')
    axes[0, 1].set_xlabel('Кластер')
    axes[0, 1].set_ylabel('Кількість оцінок')

    axes[1, 0].bar(cluster_means.index, cluster_means['unique_movies'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 0].set_title('Середня кількість унікальних фільмів')
    axes[1, 0].set_xlabel('Кластер')
    axes[1, 0].set_ylabel('Унікальні фільми')
    
    axes[1, 1].bar(cluster_means.index, cluster_means['rating_std'],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Стандартне відхилення рейтингів по кластерам')
    axes[1, 1].set_xlabel('Кластер')
    axes[1, 1].set_ylabel('Стандартне відхилення')
    
    plt.tight_layout()
    plt.show()

    print("\nІНТЕРПРЕТАЦІЯ КЛАСТЕРІВ:")
    print("=" * 40)
    
    for cluster_id in range(k):
        cluster_data = X_clustered[X_clustered['cluster'] == cluster_id]
        
        avg_rating = cluster_data['avg_rating'].mean()
        avg_count = cluster_data['ratings_count'].mean()
        avg_movies = cluster_data['unique_movies'].mean()
        avg_std = cluster_data['rating_std'].mean()
        
        print(f"\nКластер {cluster_id} (n={len(cluster_data)}):")
        print(f"  • Середній рейтинг: {avg_rating:.2f}")
        print(f"  • Середня кількість оцінок: {avg_count:.1f}")
        print(f"  • Середня кількість фільмів: {avg_movies:.1f}")
        print(f"  • Стандартне відхилення рейтингів: {avg_std:.2f}")

        if avg_count > 100 and avg_std < 1.0:
            user_type = "АКТИВНІ КОНСЕРВАТОРИ"
        elif avg_count > 100 and avg_std >= 1.0:
            user_type = "АКТИВНІ ЕКСПЕРИМЕНТАТОРИ"
        elif avg_count <= 100 and avg_rating > 3.5:
            user_type = "ВИБІРКОВІ КРИТИКИ"
        else:
            user_type = "ПАСИВНІ КОРИСТУВАЧІ"
            
        print(f"  • Тип користувача: {user_type}")
    
    return X_clustered, cluster_stats

print("\nДЕТАЛЬНИЙ АНАЛІЗ КЛАСТЕРІВ КОРИСТУВАЧІВ")
print("=" * 55)

user_data_for_clustering = user_behavior[['avg_rating', 'ratings_count', 'unique_movies', 'rating_std']]
clustered_users, stats = detailed_cluster_analysis(user_data_for_clustering, k=3)

class MovieRecommendationAnalyzer:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.movie_features = None
        self.user_features = None
        self.trained_models = {}
        
    def prepare_movie_features(self):
        print("ПІДГОТОВКА ОЗНАК ДЛЯ ФІЛЬМІВ...")

        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max'],
            'userId': 'nunique'
        }).round(3)
        
        movie_stats.columns = [
            'avg_rating', 'rating_count', 'rating_std', 
            'min_rating', 'max_rating', 'unique_users'
        ]

        movie_stats['rating_range'] = movie_stats['max_rating'] - movie_stats['min_rating']
        movie_stats['rating_stability'] = 1 - (movie_stats['rating_std'] / 5)
        movie_stats['popularity_score'] = (
            movie_stats['avg_rating'] * 
            movie_stats['rating_count'] * 
            movie_stats['rating_stability']
        )

        popularity_quantiles = movie_stats['popularity_score'].quantile([0.33, 0.66])
        movie_stats['popularity_class'] = pd.cut(
            movie_stats['popularity_score'],
            bins=[-float('inf'), popularity_quantiles[0.33], 
                  popularity_quantiles[0.66], float('inf')],
            labels=['low', 'medium', 'high']
        )
        
        self.movie_features = movie_stats
        print(f"Підготовлено ознаки для {len(movie_stats)} фільмів")
        return movie_stats
    
    def prepare_user_features(self):
        print("\nПІДГОТОВКА ОЗНАК ДЛЯ КОРИСТУВАЧІВ...")

        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).round(3)
        
        user_stats.columns = [
            'avg_rating', 'ratings_count', 'rating_std',
            'min_rating', 'max_rating', 'unique_movies'
        ]
        
        user_stats['rating_range'] = user_stats['max_rating'] - user_stats['min_rating']
        user_stats['rating_consistency'] = 1 - (user_stats['rating_std'] / 5)
        
        user_stats['rating_consistency'] = user_stats['rating_consistency'].fillna(0.5)
        
        user_stats['user_engagement'] = (
            user_stats['ratings_count'] * 
            user_stats['rating_consistency']
        )
        
        self.user_features = user_stats
        print(f"Підготовлено ознаки для {len(user_stats)} користувачів")
        return user_stats
    
    def train_popularity_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nНАВЧАННЯ КЛАСИФІКАТОРА ПОПУЛЯРНОСТІ...")

        X = self.movie_features[[
            'avg_rating', 'rating_count', 'rating_std', 'unique_users',
            'rating_range', 'rating_stability'
        ]].dropna()
        
        y = self.movie_features.loc[X.index, 'popularity_class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        train_accuracy = rf_classifier.score(X_train, y_train)
        test_accuracy = rf_classifier.score(X_test, y_test)
        
        print(f"Точність на навчальній вибірці: {train_accuracy:.3f}")
        print(f"Точність на тестовій вибірці: {test_accuracy:.3f}")
        print("\nЗвіт класифікації:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=rf_classifier.classes_,
                   yticklabels=rf_classifier.classes_)
        plt.title('Матриця помилок класифікації популярності')
        plt.ylabel('Справжні мітки')
        plt.xlabel('Прогнозовані мітки')
        plt.show()
        
        self.trained_models['popularity_classifier'] = rf_classifier
        return rf_classifier
    
    def perform_user_clustering(self, n_clusters=4):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        print(f"\nКЛАСТЕРИЗАЦІЯ КОРИСТУВАЧІВ (k={n_clusters})...")

        X = self.user_features[[
            'avg_rating', 'ratings_count', 'unique_movies', 
            'rating_consistency', 'user_engagement'
        ]].dropna()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        user_clusters = kmeans.fit_predict(X_scaled)

        user_features_clustered = self.user_features.copy()
        user_features_clustered['cluster'] = user_clusters

        silhouette_avg = silhouette_score(X_scaled, user_clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

        cluster_analysis = user_features_clustered.groupby('cluster').agg({
            'avg_rating': 'mean',
            'ratings_count': 'mean',
            'unique_movies': 'mean',
            'rating_consistency': 'mean',
            'user_engagement': 'mean'
        }).round(3)
        
        print("\nАНАЛІЗ КЛАСТЕРІВ КОРИСТУВАЧІВ:")
        print(cluster_analysis)

        self._visualize_user_clusters(user_features_clustered, X_scaled)
        
        self.trained_models['user_clusters'] = kmeans
        self.user_features = user_features_clustered
        
        return user_features_clustered
    
    def _visualize_user_clusters(self, user_data, X_scaled):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=user_data['cluster'], cmap='Set2', alpha=0.7)
        axes[0, 0].set_title('Кластеризація користувачів (PCA проекція)')
        axes[0, 0].set_xlabel('Головна компонента 1')
        axes[0, 0].set_ylabel('Головна компонента 2')
        plt.colorbar(scatter, ax=axes[0, 0])

        scatter = axes[0, 1].scatter(user_data['avg_rating'], user_data['ratings_count'],
                                   c=user_data['cluster'], cmap='Set2', alpha=0.7)
        axes[0, 1].set_title('Середній рейтинг vs Кількість оцінок')
        axes[0, 1].set_xlabel('Середній рейтинг')
        axes[0, 1].set_ylabel('Кількість оцінок')
        plt.colorbar(scatter, ax=axes[0, 1])

        cluster_activity = user_data.groupby('cluster')['user_engagement'].mean()
        axes[1, 0].bar(cluster_activity.index, cluster_activity.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 0].set_title('Середня активність по кластерам')
        axes[1, 0].set_xlabel('Кластер')
        axes[1, 0].set_ylabel('Рівень активності')

        cluster_consistency = user_data.groupby('cluster')['rating_consistency'].mean()
        axes[1, 1].bar(cluster_consistency.index, cluster_consistency.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 1].set_title('Середня узгодженість оцінок по кластерам')
        axes[1, 1].set_xlabel('Кластер')
        axes[1, 1].set_ylabel('Узгодженість оцінок')
        
        plt.tight_layout()
        plt.show()
    
    def generate_recommendation_strategy(self):
        print("\nГЕНЕРАЦІЯ СТРАТЕГІЇ РЕКОМЕНДАЦІЙ...")

        user_cluster_stats = self.user_features.groupby('cluster').agg({
            'avg_rating': 'mean',
            'ratings_count': 'mean',
            'unique_movies': 'mean',
            'rating_consistency': 'mean',
            'user_engagement': 'mean'
        }).round(3)

        strategies = {}
        
        for cluster_id in user_cluster_stats.index:
            stats = user_cluster_stats.loc[cluster_id]

            if stats['ratings_count'] > 100 and stats['rating_consistency'] > 0.8:
                strategy = {
                    'type': 'АКТИВНІ КОНСЕРВАТОРИ',
                    'recommendation': 'Схожі на високооцінені, популярні фільми',
                    'diversity': 'Низька - знайомі жанри та режисери',
                    'discovery': 'Обмежене - перевага перевірених варіантів'
                }
            elif stats['ratings_count'] > 100 and stats['rating_consistency'] <= 0.8:
                strategy = {
                    'type': 'ЕКСПЕРИМЕНТАТОРИ',
                    'recommendation': 'Різноманітні жанри, незалежне кіно',
                    'diversity': 'Висока - різні жанри та стилі',
                    'discovery': 'Високе - нові та незвичайні фільми'
                }
            elif stats['ratings_count'] > 50 and stats['ratings_count'] <= 100:
                strategy = {
                    'type': 'СТАНДАРТНІ КОРИСТУВАЧІ',
                    'recommendation': 'Збалансована суміш популярного та персоналізованого',
                    'diversity': 'Середня',
                    'discovery': 'Середнє'
                }
            else:
                strategy = {
                    'type': 'НОВІ КОРИСТУВАЧІ',
                    'recommendation': 'Популярні хіти, загальновідомі фільми',
                    'diversity': 'Середня - різні популярні жанри',
                    'discovery': 'Середнє - поступове розширення смаків'
                }
        
            strategies[cluster_id] = strategy

        print("СТРАТЕГІЇ РЕКОМЕНДАЦІЙ ДЛЯ КЛАСТЕРІВ:")
        print("=" * 60)
        
        for cluster_id, strategy in strategies.items():
            print(f"\nКластер {cluster_id} - {strategy['type']}:")
            print(f"  Рекомендації: {strategy['recommendation']}")
            print(f"  Різноманітність: {strategy['diversity']}")
            print(f"  Відкриття нового: {strategy['discovery']}")
        
        return strategies

print("ЗАПУСК КОМПЛЕКСНОГО АНАЛІЗАТОРА РЕКОМЕНДАЦІЙНОЇ СИСТЕМИ")
print("=" * 70)

analyzer = MovieRecommendationAnalyzer(movies, ratings)

movie_features = analyzer.prepare_movie_features()
user_features = analyzer.prepare_user_features()

classifier = analyzer.train_popularity_classifier()
user_clusters = analyzer.perform_user_clustering(n_clusters=4)

strategies = analyzer.generate_recommendation_strategy()

print("\n" + "="*70)
print("ВСІ АНАЛІЗИ УСПІШНО ЗАВЕРШЕНІ!")
print("="*70)

def interactive_recommendation_demo(analyzer, user_id=None):
    
    if user_id is None:
        user_id = np.random.choice(analyzer.user_features.index)
    
    print(f"\nДЕМОНСТРАЦІЯ ДЛЯ КОРИСТУВАЧА ID: {user_id}")
    print("=" * 50)

    user_data = analyzer.user_features.loc[user_id]
    user_cluster = user_data['cluster']
    
    print(f"Кластер користувача: {user_cluster}")
    print(f"Середній рейтинг: {user_data['avg_rating']:.2f}")
    print(f"Кількість оцінок: {user_data['ratings_count']:.0f}")
    print(f"Унікальні фільми: {user_data['unique_movies']:.0f}")

    user_strategy = analyzer.cached_strategies[user_cluster]
    
    print(f"\nСТРАТЕГІЯ РЕКОМЕНДАЦІЙ: {user_strategy['type']}")
    print(f"Тип рекомендацій: {user_strategy['recommendation']}")
    print(f"Рівень різноманіття: {user_strategy['diversity']}")
    print(f"Відкриття нового: {user_strategy['discovery']}")

    print(f"\nПРИКЛАДИ РЕКОМЕНДАЦІЙ:")

    popular_movies = analyzer.movie_features[
        analyzer.movie_features['popularity_class'] == 'high'
    ].nlargest(10, 'popularity_score')

    diverse_condition = analyzer.movie_features['rating_std'] > 2.0
    if diverse_condition.any():
        diverse_movies = analyzer.movie_features[diverse_condition].sample(min(10, len(analyzer.movie_features[diverse_condition])))
    else:
        diverse_condition = analyzer.movie_features['rating_std'] > 1.5
        if diverse_condition.any():
            diverse_movies = analyzer.movie_features[diverse_condition].sample(min(10, len(analyzer.movie_features[diverse_condition])))
        else:
            diverse_movies = analyzer.movie_features.sample(min(10, len(analyzer.movie_features)))

    if user_strategy['type'] == 'АКТИВНІ КОНСЕРВАТОРИ':
        recommendations = popular_movies.head(5)
        print("  - Високорейтингові популярні фільми")
    elif user_strategy['type'] == 'ЕКСПЕРИМЕНТАТОРИ':
        recommendations = diverse_movies.head(5)
        print("  - Різноманітні фільми з нестандартними рейтингами")
    else:
        mixed_recommendations = pd.concat([
            popular_movies.head(3),
            diverse_movies.head(2)
        ])
        recommendations = mixed_recommendations.sample(frac=1)
        print("  - Збалансований вибір популярного та нового")

    print(f"\nПЕРСОНАЛІЗОВАНІ РЕКОМЕНДАЦІЇ:")
    for idx, (movie_id, movie_data) in enumerate(recommendations.iterrows(), 1):
        print(f"  {idx}. Фільм ID: {movie_id}")
        print(f"     Рейтинг: {movie_data['avg_rating']:.2f}")
        print(f"     Оцінок: {movie_data['rating_count']:.0f}")
        print(f"     Стабільність: {movie_data['rating_stability']:.2f}")

print("\nІНТЕРАКТИВНА ДЕМОНСТРАЦІЯ РОБОТИ СИСТЕМИ")
print("=" * 55)

print("\nПІДГОТОВКА СТРАТЕГІЙ РЕКОМЕНДАЦІЙ...")
if not hasattr(analyzer, 'cached_strategies'):
    user_cluster_stats = analyzer.user_features.groupby('cluster').agg({
        'avg_rating': 'mean',
        'ratings_count': 'mean',
        'unique_movies': 'mean',
        'rating_consistency': 'mean',
        'user_engagement': 'mean'
    }).round(3)

    analyzer.cached_strategies = {}
    for cluster_id in user_cluster_stats.index:
        stats = user_cluster_stats.loc[cluster_id]
        
        if stats['ratings_count'] > 100 and stats['rating_consistency'] > 0.8:
            strategy = {
                'type': 'АКТИВНІ КОНСЕРВАТОРИ',
                'recommendation': 'Схожі на високооцінені, популярні фільми',
                'diversity': 'Низька - знайомі жанри та режисери',
                'discovery': 'Обмежене - перевага перевірених варіантів'
            }
        elif stats['ratings_count'] > 100 and stats['rating_consistency'] <= 0.8:
            strategy = {
                'type': 'ЕКСПЕРИМЕНТАТОРИ',
                'recommendation': 'Різноманітні жанри, незалежне кіно',
                'diversity': 'Висока - різні жанри та стилі',
                'discovery': 'Високе - нові та незвичайні фільми'
            }
        elif stats['ratings_count'] > 50 and stats['ratings_count'] <= 100:
            strategy = {
                'type': 'СТАНДАРТНІ КОРИСТУВАЧІ',
                'recommendation': 'Збалансована суміш популярного та персоналізованого',
                'diversity': 'Середня',
                'discovery': 'Середнє'
            }
        else:
            strategy = {
                'type': 'НОВІ КОРИСТУВАЧІ',
                'recommendation': 'Популярні хіти, загальновідомі фільми',
                'diversity': 'Середня - різні популярні жанри',
                'discovery': 'Середнє - поступове розширення смаків'
            }
        
        analyzer.cached_strategies[cluster_id] = strategy

print("Стратегії підготовлено!")

for i in range(3):
    interactive_recommendation_demo(analyzer)
    print("\n" + "-" * 50 + "\n")

input("Натисніть Enter для виходу...")
