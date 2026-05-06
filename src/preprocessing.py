import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import re

def process_recommendation_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    movies_path  = os.path.join(base_dir, 'data', 'movies.csv')
    ratings_path = os.path.join(base_dir, 'data', 'ratings.csv')
    
    processed_data_path = os.path.join(base_dir, 'data', 'processed_movies.csv')
    tfidf_model_path    = os.path.join(base_dir, 'ml_model', 'tfidf_vectorizer.pkl')

    try:
        movies = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)
        print(" MovieLens datasets loaded successfully.")
    except FileNotFoundError:
        print(f" Error: Files not found in {os.path.join(base_dir, 'data/')}")
        return

    movies.drop_duplicates(subset='movieId', inplace=True)
    movies.dropna(inplace=True)
    
    def clean_title(title):
        return re.sub(r'\s\(\d{4}\)', '', title).strip()
    
    movies['clean_title'] = movies['title'].apply(clean_title)

    movies['genres_features'] = movies['genres'].str.replace('|', ' ', regex=False)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres_features'])
    
    os.makedirs(os.path.dirname(tfidf_model_path), exist_ok=True)
    joblib.dump(tfidf, tfidf_model_path)
    print(f" TF-IDF Vectorization complete. Model saved at: {tfidf_model_path}")

    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'rating_count']
    
    movies = movies.merge(movie_stats, on='movieId', how='left').fillna(0)

    movies.to_csv(processed_data_path, index=False)
    print(f" Processed dataset saved at: {processed_data_path}")
    
    return movies, ratings

if __name__ == "__main__":
    process_recommendation_data()