"""
Main recommendation models for the recipe recommender system
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

class HybridRecommender:
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
        # Combine ingredients and tags for content features
        recipes_df['content'] = recipes_df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        recipes_df['content'] += ' ' + recipes_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.recipe_features = tfidf.fit_transform(recipes_df['content'])
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        
        # Get collaborative filtering scores
        user_vector = self.user_features[user_idx].reshape(1, -1)
        cf_scores = cosine_similarity(user_vector, self.recipe_features)[0]
        
        # Get content-based scores (if available)
        cb_scores = np.zeros(len(self.recipe_ids))
        if hasattr(self, 'recipe_features'):
            # For simplicity, use average content similarity
            cb_scores = np.mean(self.recipe_features.toarray(), axis=1)
        
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
        model.user_ids = model_data['user_ids']
        model.recipe_ids = model_data['recipe_ids']
        
        return model

class ContentBasedRecommender:
    """Content-based recommendation system using recipe features"""
    
    def __init__(self):
        self.tfidf_ingredients = None
        self.tfidf_tags = None
        self.recipe_similarity = None
        self.recipes_df = None
        
    def fit(self, recipes_df):
        """Train the content-based model"""
        self.recipes_df = recipes_df.copy()
        
        # TF-IDF for ingredients
        ingredients_text = recipes_df['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.tfidf_ingredients = TfidfVectorizer(max_features=500, stop_words='english')
        ingredients_matrix = self.tfidf_ingredients.fit_transform(ingredients_text)
        
        # TF-IDF for tags
        tags_text = recipes_df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.tfidf_tags = TfidfVectorizer(max_features=200, stop_words='english')
        tags_matrix = self.tfidf_tags.fit_transform(tags_text)
        
        # Combine features
        combined_matrix = np.hstack([ingredients_matrix.toarray(), tags_matrix.toarray()])
        self.recipe_similarity = cosine_similarity(combined_matrix)
        
    def recommend(self, recipe_id, n_recommendations=10):
        """Find similar recipes"""
        if recipe_id not in self.recipes_df['id'].values:
            return []
        
        recipe_idx = self.recipes_df[self.recipes_df['id'] == recipe_id].index[0]
        similar_scores = self.recipe_similarity[recipe_idx]
        
        # Get top similar recipes (excluding the input recipe)
        similar_indices = np.argsort(similar_scores)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            similar_recipe_id = self.recipes_df.iloc[idx]['id']
            similarity_score = similar_scores[idx]
            recommendations.append((similar_recipe_id, similarity_score))
            
        return recommendations

class CollaborativeRecommender:
    """Collaborative filtering using SVD"""
    
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.svd = None
        self.user_item_matrix = None
        self.user_ids = None
        self.recipe_ids = None
        
    def fit(self, interactions_df):
        """Train the collaborative filtering model"""
        # Create user-item matrix
        self.user_item_matrix = interactions_df.pivot(
            index='user_id', 
            columns='recipe_id', 
            values='rating'
        ).fillna(0)
        
        self.user_ids = self.user_item_matrix.index.tolist()
        self.recipe_ids = self.user_item_matrix.columns.tolist()
        
        # Apply SVD
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.svd.fit(self.user_item_matrix)
        
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if user_id not in self.user_ids:
            return []
        
        user_idx = self.user_ids.index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx]
        
        # Transform user ratings
        user_vector = self.svd.transform(user_ratings.values.reshape(1, -1))
        
        # Get predicted ratings
        predicted_ratings = self.svd.inverse_transform(user_vector)[0]
        
        # Get top recommendations (excluding already rated recipes)
        rated_mask = user_ratings > 0
        predicted_ratings[rated_mask] = -1  # Exclude rated recipes
        
        top_indices = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            recipe_id = self.recipe_ids[idx]
            score = predicted_ratings[idx]
            recommendations.append((recipe_id, score))
            
        return recommendations 