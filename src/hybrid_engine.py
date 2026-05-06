from src.content_model import ContentRecommender
from src.collaborative_model import CollaborativeRecommender
import pandas as pd

def get_hybrid_recommendations(user_id, movie_title, top_n=10, content_weight=0.5):
    content_rec = ContentRecommender()
    collab_rec = CollaborativeRecommender()
    
    content_results = content_rec.get_content_recommendations(movie_title, top_n=20)
    
    if isinstance(content_results, str):
        return []

    hybrid_scores = []
    
 
    for idx, row in content_results.iterrows():
        movie_id = content_rec.movies_df.loc[idx, 'movieId']
        
        content_score = 1.0
        
        predicted_rating = collab_rec.predict_rating(user_id, movie_id)
        normalized_collab = predicted_rating / 5.0
        
        final_score = (content_weight * content_score) + ((1 - content_weight) * normalized_collab)
        
        hybrid_scores.append((row['clean_title'], final_score))
    
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    
    return hybrid_scores[:top_n]