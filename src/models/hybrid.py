"""
Hybrid recommendation model combining collaborative and content-based filtering
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class HybridModel:
    """Hybrid recommendation system combining collaborative and content-based filtering"""
    
    def __init__(self, collaborative_weight=0.7, content_weight=0.3):
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.collaborative_model = None
        self.content_model = None
        self.user_item_matrix = None
        self.recipe_features = None
        self.user_features = None
        self.recipe_ids = None
        self.user_ids = None
        self.tfidf_ingredients = None
        self.tfidf_tags = None
        
    def fit(self, interactions_df, recipes_df, n_components=100):
        """Train the hybrid recommendation model"""
        print("🔄 Training Hybrid Recommendation Model...")
        
        # Prepare data
        self._prepare_data(interactions_df, recipes_df)
        
        # Train collaborative filtering (SVD)
        print("📊 Training Collaborative Filtering (SVD)...")
        self.collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_features = self.collaborative_model.fit_transform(self.user_item_matrix)
        self.recipe_features = self.collaborative_model.components_.T
        
        # Train content-based filtering
        print("📝 Training Content-Based Filtering...")
        self._train_content_based(recipes_df)
        
        print("✅ Hybrid model training complete!")
        
    def _prepare_data(self, interactions_df, recipes_df):
        """Prepare data for training"""
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot(
            index='user_id', 
            columns='recipe_id', 
            values='rating'
        ).fillna(0)
        
        # Store IDs for mapping
        self.user_ids = self.user_item_matrix.index.tolist()
        self.recipe_ids = self.user_item_matrix.columns.tolist()
        
    def _train_content_based(self, recipes_df):
        """Train content-based filtering using recipe features"""
        # TF-IDF for ingredients
        ingredients_text = recipes_df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.tfidf_ingredients = TfidfVectorizer(max_features=500, stop_words='english')
        ingredients_matrix = self.tfidf_ingredients.fit_transform(ingredients_text)
        
        # TF-IDF for tags
        tags_text = recipes_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.tfidf_tags = TfidfVectorizer(max_features=200, stop_words='english')
        tags_matrix = self.tfidf_tags.fit_transform(tags_text)
        
        # Store content features
        self.content_features = np.hstack([ingredients_matrix.toarray(), tags_matrix.toarray()])
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        
        # Get collaborative filtering scores
        user_vector = self.user_features[user_idx].reshape(1, -1)
        cf_scores = cosine_similarity(user_vector, self.recipe_features)[0]
        
        # Get content-based scores
        cb_scores = np.zeros(len(self.recipe_ids))
        if hasattr(self, 'content_features'):
            # Use average content similarity for simplicity
            cb_scores = np.mean(self.content_features, axis=1)
        
        # Combine scores
        hybrid_scores = (self.collaborative_weight * cf_scores + 
                        self.content_weight * cb_scores)
        
        # Get top recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            recipe_id = self.recipe_ids[idx]
            score = hybrid_scores[idx]
            recommendations.append((recipe_id, score))
            
        return recommendations
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'collaborative_model': self.collaborative_model,
            'user_features': self.user_features,
            'recipe_features': self.recipe_features,
            'content_features': getattr(self, 'content_features', None),
            'tfidf_ingredients': self.tfidf_ingredients,
            'tfidf_tags': self.tfidf_tags,
            'user_ids': self.user_ids,
            'recipe_ids': self.recipe_ids,
            'collaborative_weight': self.collaborative_weight,
            'content_weight': self.content_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            collaborative_weight=model_data['collaborative_weight'],
            content_weight=model_data['content_weight']
        )
        
        model.collaborative_model = model_data['collaborative_model']
        model.user_features = model_data['user_features']
        model.recipe_features = model_data['recipe_features']
        model.content_features = model_data.get('content_features')
        model.tfidf_ingredients = model_data['tfidf_ingredients']
        model.tfidf_tags = model_data['tfidf_tags']
        model.user_ids = model_data['user_ids']
        model.recipe_ids = model_data['recipe_ids']
        
        return model 