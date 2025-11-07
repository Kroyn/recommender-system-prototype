import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self):
        self.df = self._load_data()
        self.similarity_matrix = self._build_similarity_matrix()
        self.title_to_idx = {title: idx for idx, title in enumerate(self.df['title'])}
    
    def _load_data(self):
        data = {
            'title': [
                'The Dark Knight', 'The Dark Knight Rises', 'Inception', 'Interstellar',
                'Pulp Fiction', 'Reservoir Dogs', 'The Godfather', 'Fight Club',
                'The Shawshank Redemption', 'Forrest Gump'
            ],
            'features': [
                'Action Crime Drama ChristopherNolan ChristianBale HeathLedger',
                'Action Thriller ChristopherNolan ChristianBale TomHardy',
                'Action Sci-Fi Thriller ChristopherNolan LeonardoDiCaprio',
                'Sci-Fi Drama ChristopherNolan MatthewMcConaughey AnneHathaway',
                'Crime Drama QuentinTarantino JohnTravolta UmaThurman',
                'Crime Thriller QuentinTarantino HarveyKeitel TimRoth',
                'Crime Drama FrancisFordCoppola MarlonBrando AlPacino',
                'Drama DavidFincher BradPitt EdwardNorton',
                'Drama FrankDarabont TimRobbins MorganFreeman',
                'Drama Romance RobertZemeckis TomHanks'
            ]
        }
        return pd.DataFrame(data)
    
    def _build_similarity_matrix(self):
        vectorizer = TfidfVectorizer()
        feature_matrix = vectorizer.fit_transform(self.df['features'])
        return cosine_similarity(feature_matrix)
    
    def get_recommendations(self, title, n=3):
        if title not in self.title_to_idx:
            return f"Movie '{title}' not found in database."
        
        idx = self.title_to_idx[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        top_indices = [i[0] for i in sim_scores]
        
        return self.df['title'].iloc[top_indices].tolist()


if __name__ == "__main__":
    recommender = MovieRecommender()
    
    print("--- Movie Recommendation System ---\n")
    
    test_movies = ['The Dark Knight', 'Pulp Fiction', 'The Shawshank Redemption']
    
    for movie in test_movies:
        recommendations = recommender.get_recommendations(movie)
        print(f"Recommendations for '{movie}': {recommendations}")
