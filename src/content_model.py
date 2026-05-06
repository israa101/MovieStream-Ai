import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

class ContentRecommender:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.movies_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed_movies.csv'))
        self.tfidf_vectorizer = joblib.load(os.path.join(base_dir, 'ml_model', 'tfidf_vectorizer.pkl'))
        
    def get_content_recommendations(self, movie_title, top_n=10):
        tfidf_matrix = self.tfidf_vectorizer.transform(self.movies_df['genres_features'])
        
        if movie_title not in self.movies_df['clean_title'].values:
            return "Movie not found!"
            
        idx = self.movies_df[self.movies_df['clean_title'] == movie_title].index[0]
        
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        related_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
        
        return self.movies_df.iloc[related_indices][['clean_title', 'genres']]