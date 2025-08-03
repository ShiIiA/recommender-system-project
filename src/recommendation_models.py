"""
Hybrid Recommendation System
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    """Hybrid recommendation system combining collaborative and content-based filtering"""
    
    def __init__(self, collaborative_weight=0.7, content_weight=0.3):
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.collaborative_model = None
        self.user_features = None
        self.recipe_features = None
        self.content_features = None
        self.tfidf_ingredients = None
        self.user_ids = None
        self.recipe_ids = None
        self.recipe_metadata = None
        
    def fit(self, interactions_df, recipes_df, n_components=50):
        """Train the hybrid model"""
        print("Training Hybrid Recommendation Model...")
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot(
            index='user_id', columns='recipe_id', values='rating'
        ).fillna(0)
        
        self.user_ids = user_item_matrix.index.tolist()
        self.recipe_ids = user_item_matrix.columns.tolist()
        
        # Store recipe metadata
        self.recipe_metadata = recipes_df.set_index('id').to_dict('index')
        
        # Train collaborative filtering
        n_components = min(n_components, min(user_item_matrix.shape) - 1)
        self.collaborative_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_features = self.collaborative_model.fit_transform(user_item_matrix.values)
        self.recipe_features = self.collaborative_model.components_.T
        
        # Train content-based filtering
        self._train_content_based(recipes_df)
        
        print("Model training complete!")
        
    def _train_content_based(self, recipes_df):
        """Train content-based filtering"""
        # Process ingredients
        ingredients_text = recipes_df['ingredients'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x)
        )
        
        self.tfidf_ingredients = TfidfVectorizer(
            max_features=100, stop_words='english', ngram_range=(1, 1)
        )
        ingredients_matrix = self.tfidf_ingredients.fit_transform(ingredients_text)
        self.content_features = ingredients_matrix.toarray()
        
        # Store mapping for recipes
        self.content_recipe_mapping = {
            recipe_id: idx for idx, recipe_id in enumerate(recipes_df['id'])
        }
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if user_id not in self.user_ids:
            # Return popular recipes for new users
            return self._get_popular_recommendations(n_recommendations)
        
        user_idx = self.user_ids.index(user_id)
        
        # Collaborative filtering scores
        user_vector = self.user_features[user_idx].reshape(1, -1)
        cf_scores = cosine_similarity(user_vector, self.recipe_features)[0]
        
        # Content-based scores
        cb_scores = np.zeros(len(self.recipe_ids))
        if hasattr(self, 'content_features'):
            for i, recipe_id in enumerate(self.recipe_ids):
                if recipe_id in self.content_recipe_mapping:
                    content_idx = self.content_recipe_mapping[recipe_id]
                    if content_idx < len(self.content_features):
                        cb_scores[i] = np.mean(self.content_features[content_idx])
        
        # Normalize scores
        if np.max(cf_scores) > 0:
            cf_scores = cf_scores / np.max(cf_scores)
        if np.max(cb_scores) > 0:
            cb_scores = cb_scores / np.max(cb_scores)
        
        # Combine scores
        hybrid_scores = (self.collaborative_weight * cf_scores + 
                        self.content_weight * cb_scores)
        
        # Get top recommendations
        top_indices = np.argsort(hybrid_scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            recipe_id = self.recipe_ids[idx]
            score = float(hybrid_scores[idx])
            recommendations.append((recipe_id, score))
            
        return recommendations
    
    def _get_popular_recommendations(self, n_recommendations):
        """Get popular recommendations for new users"""
        popular_recipes = []
        for recipe_id in self.recipe_ids[:n_recommendations]:
            score = np.random.uniform(0.6, 0.9)  # Simulate popularity scores
            popular_recipes.append((recipe_id, score))
        return popular_recipes
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'collaborative_model': self.collaborative_model,
            'user_features': self.user_features,
            'recipe_features': self.recipe_features,
            'content_features': self.content_features,
            'tfidf_ingredients': self.tfidf_ingredients,
            'user_ids': self.user_ids,
            'recipe_ids': self.recipe_ids,
            'recipe_metadata': self.recipe_metadata,
            'content_recipe_mapping': getattr(self, 'content_recipe_mapping', {}),
            'collaborative_weight': self.collaborative_weight,
            'content_weight': self.content_weight
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            collaborative_weight=model_data['collaborative_weight'],
            content_weight=model_data['content_weight']
        )
        
        # Restore all components
        model.collaborative_model = model_data['collaborative_model']
        model.user_features = model_data['user_features']
        model.recipe_features = model_data['recipe_features']
        model.content_features = model_data['content_features']
        model.tfidf_ingredients = model_data['tfidf_ingredients']
        model.user_ids = model_data['user_ids']
        model.recipe_ids = model_data['recipe_ids']
        model.recipe_metadata = model_data['recipe_metadata']
        model.content_recipe_mapping = model_data.get('content_recipe_mapping', {})
        
        return model
