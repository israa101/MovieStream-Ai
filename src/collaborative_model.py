import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error

class CollaborativeRecommender:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(self.base_dir, 'ml_model', 'svd_model.pkl')
        self.user_item_path = os.path.join(self.base_dir, 'data', 'user_item_matrix.pkl')
        
        try:
            self.svd = joblib.load(self.model_path)
            self.matrix_reduced = joblib.load(self.user_item_path)
        except:
            self.svd = None
            self.matrix_reduced = None

    def predict_rating(self, user_id, movie_id):
      
        if self.svd is None or self.matrix_reduced is None:
            return 3.5
        
        try:
            movie_idx = movie_id % len(self.matrix_reduced)
            latent_score = np.mean(self.matrix_reduced[movie_idx]) * 5
            return min(5.0, max(1.0, latent_score + 2.5)) # تحجيم ليكون حول المتوسط
        except:
            return 3.5

    def calculate_rmse(self, test_df):
        
        if self.svd is None:
            return "N/A"
        
        actual = test_df['rating'].values
        predictions = []
        
        for _, row in test_df.iterrows():
            pred = self.predict_rating(row['userId'], row['movieId'])
            predictions.append(pred)
            
        mse = mean_squared_error(actual, predictions)
        return round(np.sqrt(mse), 3)