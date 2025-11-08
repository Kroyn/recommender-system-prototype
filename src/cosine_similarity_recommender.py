import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    encodings = ['utf-8', 'latin-1']
    
    for encoding in encodings:
        try:
            movies_df = pd.read_csv('movies.csv', encoding=encoding, on_bad_lines='skip')
            ratings_df = pd.read_csv('ratings.csv', encoding=encoding)
            users_df = pd.read_csv('users.csv', encoding=encoding)
            print(f"Data loaded successfully: {len(movies_df)} movies, {len(ratings_df)} ratings, {len(users_df)} users")
            return movies_df, ratings_df, users_df
        except:
            continue
    
    print("Failed to load data, generating synthetic dataset")
    np.random.seed(42)
    movies_df = pd.DataFrame({
        'movieId': range(1, 101),
        'title': [f'Movie {i}' for i in range(1, 101)],
        'genres': ['Action|Adventure', 'Comedy', 'Drama', 'Horror', 'Romance'] * 20
    })
    ratings_df = pd.DataFrame({
        'userId': np.random.randint(1, 101, 1000),
        'movieId': np.random.randint(1, 101, 1000),
        'rating': np.random.uniform(1, 5, 1000),
        'date': pd.date_range('2020-01-01', periods=1000, freq='D')
    })
    users_df = pd.DataFrame({
        'userId': range(1, 101),
        'age': np.random.randint(18, 60, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor'], 100)
    })
    return movies_df, ratings_df, users_df

def merge_data(ratings_df, movies_df, users_df):
    try:
        merged_df = ratings_df.merge(movies_df, on='movieId').merge(users_df, on='userId')
        print(f"Successfully merged: {len(merged_df)} records")
        return merged_df
    except Exception as e:
        print(f"Merge error: {e}, using synthetic data")
        return pd.DataFrame({
            'userId': np.random.randint(1, 101, 500),
            'movieId': np.random.randint(1, 101, 500),
            'rating': np.random.uniform(1, 5, 500),
            'genres': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror'], 500),
            'age': np.random.randint(18, 60, 500),
            'gender': np.random.choice(['M', 'F'], 500)
        })

def get_genre_preference(merged_df, user_id, genre):
    try:
        user_ratings = merged_df[merged_df['userId'] == user_id]
        if 'genres' in user_ratings.columns:
            genre_ratings = user_ratings[user_ratings['genres'].str.contains(genre, na=False)]
            if len(genre_ratings) > 0:
                return genre_ratings['rating'].mean()
    except:
        pass
    return 2.5

def create_fuzzy_system():
    age = ctrl.Antecedent(np.arange(17, 61, 1), 'age')
    genre_pref = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'genre_pref')
    recommendation = ctrl.Consequent(np.arange(0, 11, 1), 'recommendation')

    age['young'] = fuzz.trapmf(age.universe, [17, 17, 25, 30])
    age['adult'] = fuzz.trimf(age.universe, [25, 35, 45])
    age['senior'] = fuzz.trapmf(age.universe, [40, 45, 60, 60])

    genre_pref['dislike'] = fuzz.trimf(genre_pref.universe, [1, 2, 3])
    genre_pref['neutral'] = fuzz.trimf(genre_pref.universe, [2.5, 3.5, 4.5])
    genre_pref['like'] = fuzz.trimf(genre_pref.universe, [4, 4.5, 5])

    recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 2, 5])
    recommendation['medium'] = fuzz.trimf(recommendation.universe, [3, 5, 7])
    recommendation['high'] = fuzz.trimf(recommendation.universe, [5, 8, 10])

    rules = [
        ctrl.Rule(age['young'] & genre_pref['like'], recommendation['high']),
        ctrl.Rule(age['young'] & genre_pref['neutral'], recommendation['medium']),
        ctrl.Rule(age['young'] & genre_pref['dislike'], recommendation['low']),
        ctrl.Rule(age['adult'] & genre_pref['like'], recommendation['high']),
        ctrl.Rule(age['adult'] & genre_pref['neutral'], recommendation['medium']),
        ctrl.Rule(age['adult'] & genre_pref['dislike'], recommendation['low']),
        ctrl.Rule(age['senior'] & genre_pref['like'], recommendation['medium']),
        ctrl.Rule(age['senior'] & genre_pref['neutral'], recommendation['medium']),
        ctrl.Rule(age['senior'] & genre_pref['dislike'], recommendation['low'])
    ]

    recommendation_ctrl = ctrl.ControlSystem(rules)
    recommendation_system = ctrl.ControlSystemSimulation(recommendation_ctrl)
    
    return age, genre_pref, recommendation, recommendation_system

def get_recommendations(users_df, merged_df, recommendation_system, user_id, genre):
    try:
        user_data = users_df[users_df['userId'] == user_id].iloc[0]
        user_age = user_data['age']
        genre_preference = get_genre_preference(merged_df, user_id, genre)
        
        recommendation_system.input['age'] = user_age
        recommendation_system.input['genre_pref'] = genre_preference
        recommendation_system.compute()
        
        rec_score = recommendation_system.output['recommendation']
        rec_level = 'high' if rec_score >= 7 else 'medium' if rec_score >= 4 else 'low'
        
        return {
            'user_id': user_id,
            'age': user_age,
            'genre': genre,
            'genre_preference': genre_preference,
            'recommendation_score': rec_score,
            'recommendation_level': rec_level
        }
    except Exception as e:
        print(f"Error for user_id {user_id}, genre {genre}: {e}")
        return None

def visualize_membership_functions(age, genre_pref, recommendation, recommendation_system):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    for label in ['young', 'adult', 'senior']:
        plt.plot(age.universe, fuzz.interp_membership(age.universe, age[label].mf, age.universe), 
                 label=label, linewidth=2)
    plt.title('Age Membership Functions')
    plt.xlabel('Age')
    plt.ylabel('Membership')
    plt.legend()

    plt.subplot(2, 2, 2)
    for label in ['dislike', 'neutral', 'like']:
        plt.plot(genre_pref.universe, fuzz.interp_membership(genre_pref.universe, genre_pref[label].mf, genre_pref.universe), 
                 label=label, linewidth=2)
    plt.title('Genre Preference Membership Functions')
    plt.xlabel('Genre Preference')
    plt.ylabel('Membership')
    plt.legend()

    plt.subplot(2, 2, 3)
    for label in ['low', 'medium', 'high']:
        plt.plot(recommendation.universe, fuzz.interp_membership(recommendation.universe, recommendation[label].mf, recommendation.universe), 
                 label=label, linewidth=2)
    plt.title('Recommendation Membership Functions')
    plt.xlabel('Recommendation Level')
    plt.ylabel('Membership')
    plt.legend()

    plt.subplot(2, 2, 4)
    age_range = np.arange(17, 61, 2)
    pref_range = np.arange(1, 5.1, 0.2)
    X, Y = np.meshgrid(age_range, pref_range)
    Z = np.zeros(X.shape)

    for i in range(len(age_range)):
        for j in range(len(pref_range)):
            recommendation_system.input['age'] = age_range[i]
            recommendation_system.input['genre_pref'] = pref_range[j]
            try:
                recommendation_system.compute()
                Z[j, i] = recommendation_system.output['recommendation']
            except:
                Z[j, i] = 0

    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Recommendation Level')
    plt.xlabel('Age')
    plt.ylabel('Genre Preference')
    plt.title('Fuzzy System Decision Surface')

    plt.tight_layout()
    plt.savefig('fuzzy_system_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_recommendations(users_df, merged_df, recommendation_system, test_genres):
    print("\nAnalyzing recommendations for selected users:")
    print("=" * 60)

    available_users = users_df['userId'].unique()[:5]
    results = []
    
    for user_id in available_users:
        user_results = []
        for genre in test_genres:
            result = get_recommendations(users_df, merged_df, recommendation_system, user_id, genre)
            if result is not None:
                user_results.append(result)
                results.append(result)
        
        if user_results:
            print(f"\nUser {user_id} (age: {user_results[0]['age']}):")
            for res in user_results:
                print(f"  {res['genre']}: preference {res['genre_preference']:.2f} -> "
                      f"recommendation {res['recommendation_score']:.2f} ({res['recommendation_level']})")
    
    return results

def visualize_results(results_df, users_df, test_genres):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    pivot_table = results_df.pivot_table(values='recommendation_score', 
                                       index='user_id', 
                                       columns='genre', 
                                       aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Recommendation Levels Heatmap')
    plt.xlabel('Genre')
    plt.ylabel('User ID')

    plt.subplot(2, 2, 2)
    scatter = plt.scatter(results_df['age'], results_df['recommendation_score'], 
                         c=results_df['genre_preference'], cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter, label='Genre Preference')
    plt.xlabel('Age')
    plt.ylabel('Recommendation Level')
    plt.title('Recommendations Distribution by Age')

    plt.subplot(2, 2, 3)
    genre_avg = results_df.groupby('genre')['recommendation_score'].mean()
    plt.bar(genre_avg.index, genre_avg.values, color=['red', 'blue', 'green'])
    plt.title('Average Recommendation Level by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Recommendation Level')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    plt.hist(users_df['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Number of Users')
    plt.title('User Age Distribution')

    plt.tight_layout()
    plt.savefig('recommendation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_3d(results_df, test_genres):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Action': 'red', 'Comedy': 'blue', 'Drama': 'green'}
    for genre in test_genres:
        genre_data = results_df[results_df['genre'] == genre]
        ax.scatter(genre_data['age'], genre_data['genre_preference'], genre_data['recommendation_score'],
                  c=colors[genre], label=genre, s=100, alpha=0.7)

    ax.set_xlabel('Age')
    ax.set_ylabel('Genre Preference')
    ax.set_zlabel('Recommendation Level')
    ax.set_title('3D Recommendation System Visualization')
    ax.legend()

    plt.savefig('3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(users_df, movies_df, ratings_df):
    print("\n" + "=" * 60)
    print("SYSTEM STATISTICS:")
    print("=" * 60)
    print(f"Total users: {len(users_df)}")
    print(f"Total movies: {len(movies_df)}")
    print(f"Total ratings: {len(ratings_df)}")
    print(f"Average user age: {users_df['age'].mean():.1f}")
    if 'rating' in ratings_df.columns:
        print(f"Average movie rating: {ratings_df['rating'].mean():.2f}")

def demonstrate_new_user(recommendation_system):
    print("\n" + "=" * 60)
    print("NEW USER RECOMMENDATION EXAMPLE:")
    print("=" * 60)

    new_user_age = 28
    new_user_genre_pref = 4.5

    recommendation_system.input['age'] = new_user_age
    recommendation_system.input['genre_pref'] = new_user_genre_pref
    recommendation_system.compute()

    recommendation_score = recommendation_system.output['recommendation']
    recommendation_level = 'high' if recommendation_score >= 7 else 'medium' if recommendation_score >= 4 else 'low'

    print(f"Age: {new_user_age}")
    print(f"Genre preference: {new_user_genre_pref}")
    print(f"Recommendation level: {recommendation_score:.2f} ({recommendation_level})")

    print(f"\nRecommendations for age {new_user_age}:")
    if recommendation_level == 'high':
        print("  - Action movies (The Dark Knight, Inception)")
        print("  - Adventure movies (Indiana Jones, Jurassic Park)")
    elif recommendation_level == 'medium':
        print("  - Comedies (Toy Story, Shrek)")
        print("  - Dramas (Forrest Gump, The Shawshank Redemption)")
    else:
        print("  - Classic movies (The Godfather, Pulp Fiction)")
        print("  - Art house cinema")

def main():
    movies_df, ratings_df, users_df = load_data()
    merged_df = merge_data(ratings_df, movies_df, users_df)
    
    print("\nInitializing fuzzy system...")
    age, genre_pref, recommendation, recommendation_system = create_fuzzy_system()
    
    print("\nCreating visualizations...")
    visualize_membership_functions(age, genre_pref, recommendation, recommendation_system)
    
    test_genres = ['Action', 'Comedy', 'Drama']
    results = analyze_recommendations(users_df, merged_df, recommendation_system, test_genres)
    
    if results:
        results_df = pd.DataFrame(results)
        visualize_results(results_df, users_df, test_genres)
        visualize_3d(results_df, test_genres)
    
    print_statistics(users_df, movies_df, ratings_df)
    demonstrate_new_user(recommendation_system)
    
    print("\nComplete! Visualizations saved to:")
    print("- fuzzy_system_visualization.png")
    print("- recommendation_analysis.png") 
    print("- 3d_visualization.png")

if __name__ == "__main__":
    main()
