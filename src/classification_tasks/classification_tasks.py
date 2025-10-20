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
from sklearn.metrics import confusion_matrix

# --- Global Variables ---
movies_df = None
ratings_df = None
movie_stats = None
user_behavior = None
analyzer = None

def load_data():
    global movies_df, ratings_df, movie_stats, user_behavior
    print("Loading data...")
    try:
        # Try to find the script directory (works in most environments)
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for environments where __file__ is not defined (like some notebooks)
            script_dir = os.getcwd()

        movies_path = os.path.join(script_dir, 'movies.csv')
        ratings_path = os.path.join(script_dir, 'ratings.csv')

        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        print("Data loaded successfully")
        print(f"Movies: {len(movies_df)}, Ratings: {len(ratings_df)}")

        prepare_movie_features()
        prepare_user_features()
        print("Features prepared successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'movies.csv' and 'ratings.csv' are in the same directory as the script.")
        return False
    return True

def prepare_movie_features():
    global movie_stats
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count', 'std'],
        'userId': 'nunique'
    }).round(3)
    movie_stats.columns = ['avg_rating', 'rating_count', 'rating_std', 'unique_users']

    movie_stats['popularity_score'] = (
        movie_stats['avg_rating'] *
        movie_stats['rating_count'] *
        (1 - movie_stats['rating_std'].fillna(0)/5) # Handle potential NaN in std
    )

    threshold_upper = movie_stats['popularity_score'].quantile(0.75)
    threshold_lower = movie_stats['popularity_score'].quantile(0.25)

    movie_stats['popularity_class'] = pd.cut(
        movie_stats['popularity_score'],
        bins=[-float('inf'), threshold_lower, threshold_upper, float('inf')],
        labels=['low', 'medium', 'high']
    )

    for column in ['avg_rating', 'rating_count']:
        movie_stats = handle_outliers_iqr(movie_stats, column)

def prepare_user_features():
    global user_behavior
    user_behavior = ratings_df.groupby('userId').agg({
        'rating': ['mean', 'count', 'std'],
        'movieId': 'nunique'
    }).round(3)
    user_behavior.columns = ['avg_rating', 'ratings_count', 'rating_std', 'unique_movies']
    user_behavior = user_behavior.dropna()


def prepare_data(X, y):
    common_index = X.dropna().index.intersection(y.dropna().index)
    X_clean = X.loc[common_index]
    y_clean = y.loc[common_index]
    return X_clean, y_clean

def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Outliers in {column}: {len(outliers)} records")

    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def task_1_data_quality_report():
    print("\n--- Task 1: Data Quality Report & Feature Info ---")
    if movie_stats is None:
        print("Data not loaded. Please load data first.")
        return

    print("\nDistribution of movies by popularity:")
    print(movie_stats['popularity_class'].value_counts())

    X = movie_stats[['avg_rating', 'rating_count', 'rating_std', 'unique_users']]
    y = movie_stats['popularity_class']

    X, y = prepare_data(X, y)
    print(f"\nData dimensions after cleaning: X{X.shape}, y{y.shape}")
    print("\nDetailed class distribution:")
    print(y.value_counts())

    print("\nDATA QUALITY REPORT")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Missing in X: {X.isna().sum().sum()}")
    print(f"Missing in y: {y.isna().sum()}")
    print(f"Class Balance:")
    print(y.value_counts(normalize=True).round(3))
    print(f"Minimum class count: {y.value_counts().min()}")

    mi_scores = mutual_info_classif(X, y, random_state=42)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)

    print("\nFeature Informativeness (Mutual Info):")
    print(feature_importance)

def task_2_classification_types_demo():
    print("\n--- Task 2: Classification Types Demo ---")

    print("\n=== BINARY CLASSIFICATION ===")
    movie_stats['is_highly_rated'] = (movie_stats['avg_rating'] >= 4.0).astype(int)
    X_binary = movie_stats[['rating_count', 'unique_users', 'rating_std']]
    y_binary = movie_stats['is_highly_rated']
    X_binary, y_binary = prepare_data(X_binary, y_binary)

    model_bin = LogisticRegression(random_state=42)
    model_bin.fit(X_binary, y_binary)
    accuracy_bin = model_bin.score(X_binary, y_binary)
    print(f"Binary Classification Accuracy: {accuracy_bin:.3f}")

    print("\n=== MULTICLASS CLASSIFICATION ===")
    X_multi = movie_stats[['rating_count', 'unique_users', 'rating_std', 'avg_rating']]
    y_multi = movie_stats['popularity_class']
    X_multi, y_multi = prepare_data(X_multi, y_multi)

    model_multi = RandomForestClassifier(n_estimators=100, random_state=42)
    model_multi.fit(X_multi, y_multi)
    accuracy_multi = model_multi.score(X_multi, y_multi)
    print(f"Multiclass Classification Accuracy: {accuracy_multi:.3f}")

def task_3_multilabel_demo():
    print("\n--- Task 3: Multi-Label Classification Demo ---")
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

    print(f"\nData dimensions for multi-label classification: {X_ml.shape}")
    print("Genre distribution:")
    print(y_ml.sum().sort_values(ascending=False))

def task_4_algorithm_comparison():
    print("\n--- Task 4: Comparison of Classification Algorithms ---")

    X_compare = movie_stats[['avg_rating', 'rating_count', 'rating_std', 'unique_users']]
    y_compare = movie_stats['popularity_class']
    X_compare, y_compare = prepare_data(X_compare, y_compare)

    if X_compare.empty:
        print("Not enough data to compare algorithms.")
        return

    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-NN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_compare)
    results = {}
    min_class_size = y_compare.value_counts().min()
    n_splits = min(5, min_class_size)

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"Using Stratified {n_splits}-Fold Cross-Validation")
    else:
        print("Using train-test split due to small class size")
        cv = None

    for name, model in algorithms.items():
        start_time = time.time()
        try:
            use_scaled = name in ['SVM', 'K-NN', 'Logistic Regression', 'Naive Bayes']
            X_data = X_scaled if use_scaled else X_compare

            if cv:
                scores = cross_val_score(model, X_data, y_compare, cv=cv, scoring='accuracy')
                mean_accuracy = scores.mean()
                std_accuracy = scores.std()
                method = f"CV ({n_splits}-fold)"
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_data, y_compare, test_size=0.3, random_state=42, stratify=y_compare
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
            results[name] = {'mean_accuracy': 0, 'std_accuracy': 0, 'time_seconds': 0, 'method': 'failed'}

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

            plt.title('Comparison of Classification Algorithm Accuracy', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy', fontsize=12)
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
            print(f"\nBest Algorithm: {best_model} with accuracy {best_accuracy:.3f}")

def task_5_random_forest_deep_dive():
    print("\n--- Task 5: Random Forest Deep Dive ---")

    X_compare = movie_stats[['avg_rating', 'rating_count', 'rating_std', 'unique_users']]
    y_compare = movie_stats['popularity_class']
    X, y = prepare_data(X_compare, y_compare)

    if X.empty or len(X) < 10:
        print("Not enough data for a deep dive analysis.")
        return

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    n_samples = len(X)
    train_sizes_abs = np.linspace(0.1, 1.0, 5) * n_samples
    train_sizes_abs = train_sizes_abs.astype(int)
    train_sizes_abs = np.unique(train_sizes_abs)

    min_class_size = y.value_counts().min()
    cv_folds = min(5, min_class_size)
    if cv_folds < 2:
        cv_folds = 2

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            rf, X, y, cv=cv_folds, scoring='accuracy',
            train_sizes=train_sizes_abs,
            random_state=42, n_jobs=-1
        )

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Set')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='Test Set')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.title('Random Forest Learning Curve')
        plt.legend()
        plt.grid(True)
    except Exception as e:
        print(f"Could not generate learning curve: {e}")

    rf.fit(X, y)
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.subplot(1, 2, 2)
    plt.barh(feature_imp['feature'], feature_imp['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Random Forest')
    plt.tight_layout()
    plt.show()

    print("\nFeature Importance from Model:")
    print(feature_importance)

def task_6_classification_vs_clustering():
    print("\n--- Task 6: Classification vs. Clustering Comparison ---")

    X_comp = movie_stats[['avg_rating', 'rating_count', 'unique_users']].dropna()
    y_comp = movie_stats.loc[X_comp.index, 'popularity_class']
    label_map = {'low': 0, 'medium': 1, 'high': 2}
    y_comp_numeric = y_comp.map(label_map)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_comp)

    clustering_algorithms = {
        'K-Means': KMeans(n_clusters=3, random_state=42, n_init=10),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=3)
    }
    results = {}

    for name, algorithm in clustering_algorithms.items():
        print(f"\n--- {name} ---")
        labels = algorithm.fit_predict(X_scaled)

        if name == 'DBSCAN':
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"Number of clusters: {n_clusters}")
            print(f"Noise points: {n_noise}")
        else:
            n_clusters = len(set(labels))
            print(f"Number of clusters: {n_clusters}")

        if n_clusters > 1:
            silhouette = silhouette_score(X_scaled, labels)
            calinski = calinski_harabasz_score(X_scaled, labels)
            davies = davies_bouldin_score(X_scaled, labels)

            print(f"Silhouette Score: {silhouette:.3f}")
            print(f"Calinski-Harabasz: {calinski:.3f}")
            print(f"Davies-Bouldin: {davies:.3f}")

            results[name] = {
                'labels': labels, 'silhouette': silhouette,
                'calinski': calinski, 'davies': davies,
                'n_clusters': n_clusters
            }

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_comp_numeric, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('True Classes (Classification)')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[0, 0])

    plot_idx = 1
    for name, result in results.items():
        row, col = plot_idx // 2, plot_idx % 2
        scatter = axes[row, col].scatter(X_pca[:, 0], X_pca[:, 1], c=result['labels'], cmap='Set2', alpha=0.7)
        axes[row, col].set_title(f'{name}\nSilhouette: {result["silhouette"]:.3f}')
        axes[row, col].set_xlabel('PC1')
        axes[row, col].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[row, col])
        plot_idx += 1

    if plot_idx <= 3:
        fig.delaxes(axes[1][1])

    plt.tight_layout()
    plt.show()

    applications = {
        'Classification': {
            'Task': 'Predicting movie popularity',
            'Input Data': 'Historical ratings, movie metadata',
            'Result': 'Labels "low", "medium", "high" popularity',
            'Usage': 'Content filtering, recommendation prioritization',
            'Pros': 'Clear interpretable results, high accuracy',
            'Cons': 'Requires pre-labeled data'
        },
        'Clustering': {
            'Task': 'User segmentation by behavior',
            'Input Data': 'Viewing patterns, ratings, activity frequency',
            'Result': 'Groups of users with similar tastes',
            'Usage': 'Personalization of recommendations, targeted marketing',
            'Pros': 'Does not require labels, reveals hidden patterns',
            'Cons': 'Difficulty in interpretation, dependency on parameters'
        }
    }
    comparison_df = pd.DataFrame(applications).T
    print("\nCOMPARISON TABLE: CLASSIFICATION vs CLUSTERING")
    print("=" * 80)
    print(comparison_df)

def task_7_clustering_quality_evaluation():
    print("\n--- Task 7: Clustering Quality Evaluation (Elbow, Silhouette) ---")
    if user_behavior is None or len(user_behavior) < 10:
        print("Not enough user data to evaluate clustering.")
        return 3 # Return a default k

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_behavior)

    max_k = min(10, len(X_scaled) - 1)
    if max_k < 2:
        print("Not enough data points to perform clustering evaluation.")
        return 3 # Return a default k

    k_range = range(2, max_k + 1)
    metrics = {'inertia': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X_scaled, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
        metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(k_range, metrics['inertia'], 'bo-')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].set_xlabel('Number of clusters (k)')
    axes[0, 0].set_ylabel('Inertia')

    axes[0, 1].plot(k_range, metrics['silhouette'], 'go-')
    axes[0, 1].set_title('Silhouette Score')
    axes[0, 1].set_xlabel('Number of clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')

    axes[1, 0].plot(k_range, metrics['calinski_harabasz'], 'ro-')
    axes[1, 0].set_title('Calinski-Harabasz Index')
    axes[1, 0].set_xlabel('Number of clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz')

    axes[1, 1].plot(k_range, metrics['davies_bouldin'], 'mo-')
    axes[1, 1].set_title('Davies-Bouldin Index')
    axes[1, 1].set_xlabel('Number of clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin')

    plt.tight_layout()
    plt.show()

    print("\nOPTIMAL NUMBER OF CLUSTERS:")
    optimal_k_silhouette = k_range[np.argmax(metrics['silhouette'])]
    print(f"By Silhouette Score: k = {optimal_k_silhouette}")
    optimal_k_calinski = k_range[np.argmax(metrics['calinski_harabasz'])]
    print(f"By Calinski-Harabasz: k = {optimal_k_calinski}")
    optimal_k_davies = k_range[np.argmin(metrics['davies_bouldin'])]
    print(f"By Davies-Bouldin: k = {optimal_k_davies}")

    return optimal_k_silhouette

def task_8_detailed_cluster_analysis():
    print("\n--- Task 8: Detailed User Cluster Analysis ---")
    if user_behavior is None or len(user_behavior) < 3:
        print("Not enough user data for cluster analysis.")
        return

    user_data_for_clustering = user_behavior[['avg_rating', 'ratings_count', 'unique_movies', 'rating_std']]

    k = 3 # Default k
    if len(user_data_for_clustering) >= 10:
        print("First, determining optimal k...")
        k = task_7_clustering_quality_evaluation()
        print(f"\nUsing optimal k={k} for detailed analysis.")
    else:
         print(f"Using default k={k} as dataset is small.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_data_for_clustering)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    X_clustered = user_data_for_clustering.copy()
    X_clustered['cluster'] = labels

    cluster_stats = X_clustered.groupby('cluster').agg({
        'avg_rating': ['mean', 'std', 'count'],
        'ratings_count': ['mean', 'std'],
        'unique_movies': ['mean', 'std'],
        'rating_std': ['mean', 'std']
    }).round(3)

    print("\nCLUSTER STATISTICS:")
    print("=" * 50)
    print(cluster_stats)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    cluster_means = X_clustered.groupby('cluster').mean()
    colors = plt.cm.Set2(np.linspace(0, 1, k))

    axes[0, 0].bar(cluster_means.index, cluster_means['avg_rating'], color=colors)
    axes[0, 0].set_title('Average Rating by Cluster')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Average Rating')

    axes[0, 1].bar(cluster_means.index, cluster_means['ratings_count'], color=colors)
    axes[0, 1].set_title('Average Rating Count by Cluster')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Rating Count')

    axes[1, 0].bar(cluster_means.index, cluster_means['unique_movies'], color=colors)
    axes[1, 0].set_title('Average Unique Movies by Cluster')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Unique Movies')

    axes[1, 1].bar(cluster_means.index, cluster_means['rating_std'], color=colors)
    axes[1, 1].set_title('Rating Std Dev by Cluster')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Rating Std Dev')

    plt.tight_layout()
    plt.show()

    print("\nCLUSTER INTERPRETATION:")
    print("=" * 40)
    for cluster_id in range(k):
        cluster_data = X_clustered[X_clustered['cluster'] == cluster_id]
        avg_rating = cluster_data['avg_rating'].mean()
        avg_count = cluster_data['ratings_count'].mean()
        avg_std = cluster_data['rating_std'].mean()

        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
        print(f"  • Avg Rating: {avg_rating:.2f}")
        print(f"  • Avg Count: {avg_count:.1f}")
        print(f"  • Avg Std Dev: {avg_std:.2f}")

        if avg_count > 100 and avg_std < 1.0:
            user_type = "ACTIVE CONSERVATIVES"
        elif avg_count > 100 and avg_std >= 1.0:
            user_type = "ACTIVE EXPLORERS"
        elif avg_count <= 100 and avg_rating > 3.5:
            user_type = "SELECTIVE CRITICS"
        else:
            user_type = "CASUAL USERS"
        print(f"  • User Type: {user_type}")

class MovieRecommendationAnalyzer:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.movie_features = None
        self.user_features = None
        self.trained_models = {}
        self.cached_strategies = {}

    def prepare_movie_features(self):
        print("PREPARING MOVIE FEATURES...")
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max'],
            'userId': 'nunique'
        }).round(3)
        movie_stats.columns = [
            'avg_rating', 'rating_count', 'rating_std',
            'min_rating', 'max_rating', 'unique_users'
        ]
        movie_stats['rating_range'] = movie_stats['max_rating'] - movie_stats['min_rating']
        movie_stats['rating_stability'] = 1 - (movie_stats['rating_std'].fillna(0) / 5)
        movie_stats['popularity_score'] = (
            movie_stats['avg_rating'] * movie_stats['rating_count'] * movie_stats['rating_stability']
        )
        popularity_quantiles = movie_stats['popularity_score'].quantile([0.33, 0.66])
        movie_stats['popularity_class'] = pd.cut(
            movie_stats['popularity_score'],
            bins=[-float('inf'), popularity_quantiles[0.33],
                  popularity_quantiles[0.66], float('inf')],
            labels=['low', 'medium', 'high']
        )
        self.movie_features = movie_stats.dropna()
        print(f"Prepared features for {len(self.movie_features)} movies")

    def prepare_user_features(self):
        print("\nPREPARING USER FEATURES...")
        user_stats = self.ratings_df.groupby('userId').agg({
            'rating': ['mean', 'count', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).round(3)
        user_stats.columns = [
            'avg_rating', 'ratings_count', 'rating_std',
            'min_rating', 'max_rating', 'unique_movies'
        ]
        user_stats['rating_range'] = user_stats['max_rating'] - user_stats['min_rating']
        user_stats['rating_consistency'] = 1 - (user_stats['rating_std'].fillna(0) / 5)
        user_stats['rating_consistency'] = user_stats['rating_consistency'].fillna(0.5)
        user_stats['user_engagement'] = (
            user_stats['ratings_count'] * user_stats['rating_consistency']
        )
        self.user_features = user_stats.dropna()
        print(f"Prepared features for {len(self.user_features)} users")

    def train_popularity_classifier(self):
        print("\nTRAINING POPULARITY CLASSIFIER...")
        X = self.movie_features[[
            'avg_rating', 'rating_count', 'rating_std', 'unique_users',
            'rating_range', 'rating_stability'
        ]]
        y = self.movie_features['popularity_class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        print(f"Training Accuracy: {rf_classifier.score(X_train, y_train):.3f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred, labels=rf_classifier.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=rf_classifier.classes_,
                   yticklabels=rf_classifier.classes_)
        plt.title('Popularity Classification Confusion Matrix')
        plt.ylabel('True Labels')
        plt.xlabel('Predicted Labels')
        plt.show()

        self.trained_models['popularity_classifier'] = rf_classifier

    def perform_user_clustering(self, n_clusters=4):
        print(f"\nPERFORMING USER CLUSTERING (k={n_clusters})...")
        X = self.user_features[[
            'avg_rating', 'ratings_count', 'unique_movies',
            'rating_consistency', 'user_engagement'
        ]]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_clusters = kmeans.fit_predict(X_scaled)

        self.user_features['cluster'] = user_clusters
        silhouette_avg = silhouette_score(X_scaled, user_clusters)
        print(f"Silhouette Score: {silhouette_avg:.3f}")

        cluster_analysis = self.user_features.groupby('cluster').mean(numeric_only=True).round(3)
        print("\nUSER CLUSTER ANALYSIS:")
        print(cluster_analysis)

        self._visualize_user_clusters(X_scaled, user_clusters)
        self.trained_models['user_clusters'] = kmeans

    def _visualize_user_clusters(self, X_scaled, labels):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        scatter = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set2', alpha=0.7)
        axes[0, 0].set_title('User Clustering (PCA Projection)')
        axes[0, 0].set_xlabel('Principal Component 1')
        axes[0, 0].set_ylabel('Principal Component 2')
        plt.colorbar(scatter, ax=axes[0, 0])

        scatter = axes[0, 1].scatter(self.user_features['avg_rating'], self.user_features['ratings_count'],
                                   c=labels, cmap='Set2', alpha=0.7)
        axes[0, 1].set_title('Avg Rating vs Rating Count')
        axes[0, 1].set_xlabel('Average Rating')
        axes[0, 1].set_ylabel('Rating Count')
        plt.colorbar(scatter, ax=axes[0, 1])

        cluster_activity = self.user_features.groupby('cluster')['user_engagement'].mean()
        axes[1, 0].bar(cluster_activity.index, cluster_activity.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 0].set_title('Average Engagement by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Engagement Level')

        cluster_consistency = self.user_features.groupby('cluster')['rating_consistency'].mean()
        axes[1, 1].bar(cluster_consistency.index, cluster_consistency.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[1, 1].set_title('Average Rating Consistency by Cluster')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Rating Consistency')

        plt.tight_layout()
        plt.show()

    def generate_recommendation_strategy(self):
        print("\nGENERATING RECOMMENDATION STRATEGIES...")
        user_cluster_stats = self.user_features.groupby('cluster').mean(numeric_only=True)
        strategies = {}

        for cluster_id in user_cluster_stats.index:
            stats = user_cluster_stats.loc[cluster_id]
            if stats['ratings_count'] > 100 and stats['rating_consistency'] > 0.8:
                strategy = {
                    'type': 'ACTIVE CONSERVATIVES',
                    'recommendation': 'Similar to highly-rated, popular films',
                    'diversity': 'Low - familiar genres and directors',
                    'discovery': 'Limited - preference for proven options'
                }
            elif stats['ratings_count'] > 100 and stats['rating_consistency'] <= 0.8:
                strategy = {
                    'type': 'EXPLORERS',
                    'recommendation': 'Diverse genres, independent cinema',
                    'diversity': 'High - different genres and styles',
                    'discovery': 'High - new and unusual films'
                }
            elif 50 < stats['ratings_count'] <= 100:
                strategy = {
                    'type': 'STANDARD USERS',
                    'recommendation': 'Balanced mix of popular and personalized',
                    'diversity': 'Medium',
                    'discovery': 'Medium'
                }
            else:
                strategy = {
                    'type': 'NEW/CASUAL USERS',
                    'recommendation': 'Popular hits, well-known films',
                    'diversity': 'Medium - various popular genres',
                    'discovery': 'Medium - gradual expansion of tastes'
                }
            strategies[cluster_id] = strategy

        print("RECOMMENDATION STRATEGIES FOR CLUSTERS:")
        print("=" * 60)
        for cluster_id, strategy in strategies.items():
            print(f"\nCluster {cluster_id} - {strategy['type']}:")
            print(f"  Recommendation: {strategy['recommendation']}")
            print(f"  Diversity: {strategy['diversity']}")
            print(f"  Discovery: {strategy['discovery']}")

        self.cached_strategies = strategies

def task_9_run_full_recommendation_analyzer():
    global analyzer
    print("\n--- Task 9: Run Full Recommendation System Analyzer ---")
    analyzer = MovieRecommendationAnalyzer(movies_df, ratings_df)
    analyzer.prepare_movie_features()
    analyzer.prepare_user_features()
    analyzer.train_popularity_classifier()
    analyzer.perform_user_clustering(n_clusters=4)
    analyzer.generate_recommendation_strategy()
    print("\n" + "="*70)
    print("ALL ANALYSES SUCCESSFULLY COMPLETED!")
    print("="*70)

def task_10_interactive_demo():
    print("\n--- Task 10: Interactive Recommendation Demo ---")
    if analyzer is None or not analyzer.cached_strategies:
        print("Error: The full analyzer (Task 9) must be run first to generate strategies.")
        return

    for i in range(3):
        user_id = np.random.choice(analyzer.user_features.index)
        print(f"\nDEMO FOR USER ID: {user_id}")
        print("=" * 50)

        user_data = analyzer.user_features.loc[user_id]
        user_cluster = int(user_data['cluster'])

        print(f"User Cluster: {user_cluster}")
        print(f"Avg Rating: {user_data['avg_rating']:.2f}")
        print(f"Rating Count: {user_data['ratings_count']:.0f}")
        print(f"Unique Movies: {user_data['unique_movies']:.0f}")

        user_strategy = analyzer.cached_strategies[user_cluster]

        print(f"\nRECOMMENDATION STRATEGY: {user_strategy['type']}")
        print(f"Recommendation Type: {user_strategy['recommendation']}")
        print(f"Diversity Level: {user_strategy['diversity']}")
        print(f"Discovery Level: {user_strategy['discovery']}")

        print(f"\nEXAMPLE RECOMMENDATIONS:")

        popular_movies = analyzer.movie_features[
            analyzer.movie_features['popularity_class'] == 'high'
        ].nlargest(10, 'popularity_score')

        diverse_condition = analyzer.movie_features['rating_std'] > 1.0
        if diverse_condition.sum() < 5:
             diverse_condition = analyzer.movie_features['rating_std'] > 0.5

        diverse_movies = analyzer.movie_features[diverse_condition].sample(
            min(10, diverse_condition.sum())
        )

        if user_strategy['type'] == 'ACTIVE CONSERVATIVES':
            recommendations = popular_movies.head(5)
            print("  - Highly-rated popular films")
        elif user_strategy['type'] == 'EXPLORERS':
            recommendations = diverse_movies.head(5)
            print("  - Diverse films with non-standard ratings")
        else:
            mixed_recommendations = pd.concat([
                popular_movies.head(3),
                diverse_movies.head(2)
            ]).drop_duplicates()
            recommendations = mixed_recommendations.sample(frac=1).head(5)
            print("  - Balanced mix of popular and new")

        print(f"\nPERSONALIZED RECOMMENDATIONS:")
        for idx, (movie_id, movie_data) in enumerate(recommendations.iterrows(), 1):
            print(f"  {idx}. Movie ID: {movie_id}")
            print(f"     Rating: {movie_data['avg_rating']:.2f}")
            print(f"     Count: {movie_data['rating_count']:.0f}")
            print(f"     Stability: {movie_data['rating_stability']:.2f}")

        print("\n" + "-" * 50 + "\n")

def main_menu():
    if not load_data():
        return

    while True:
        print("\n" + "=" * 50)
        print("    Classification and Clustering Main Menu")
        print("=" * 50)
        print("1. Data Quality & Feature Report")
        print("2. Classification Types Demo (Binary/Multiclass)")
        print("3. Multi-Label Classification Demo")
        print("4. Compare Classification Algorithms")
        print("5. Random Forest Deep Dive (Learning Curve, Features)")
        print("6. Classification vs. Clustering Comparison (Theory & Viz)")
        print("7. Clustering Quality Evaluation (Elbow, Silhouette)")
        print("8. Detailed User Cluster Analysis & Interpretation")
        print("---")
        print("9. Run Full Recommendation System Analyzer (Tasks 1-8)")
        print("10. Run Interactive Recommendation Demo (Requires Task 9)")
        print("0. Exit")

        choice = input("Enter your choice (0-10): ")

        if choice == '1':
            task_1_data_quality_report()
        elif choice == '2':
            task_2_classification_types_demo()
        elif choice == '3':
            task_3_multilabel_demo()
        elif choice == '4':
            task_4_algorithm_comparison()
        elif choice == '5':
            task_5_random_forest_deep_dive()
        elif choice == '6':
            task_6_classification_vs_clustering()
        elif choice == '7':
            task_7_clustering_quality_evaluation()
        elif choice == '8':
            task_8_detailed_cluster_analysis()
        elif choice == '9':
            task_9_run_full_recommendation_analyzer()
        elif choice == '10':
            task_10_interactive_demo()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number from 0 to 10.")

        if choice != '0':
            input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    # Add a warning about KMeans n_init
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
    main_menu()
