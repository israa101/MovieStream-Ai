import pandas as pd
from collaborative_model import CollaborativeRecommender
from sklearn.decomposition import TruncatedSVD
import joblib
import os

def train():
    ratings = pd.read_csv('data/ratings.csv')
    
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_reduced = svd.fit_transform(user_item_matrix.T) 
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'ml_model', 'svd_model.pkl')
    matrix_path = os.path.join(base_dir, 'data', 'user_item_matrix.pkl')
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    joblib.dump(svd, model_path)
    joblib.dump(matrix_reduced, matrix_path)
    
    print(f" SVD Model saved at: {model_path}")
    print(f" User-Item Matrix saved at: {matrix_path}")

if __name__ == "__main__":
    train()