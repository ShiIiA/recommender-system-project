"""
Utility functions for the recipe recommender system
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ast
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        return None

def save_pickle(data: Any, filepath: Path) -> bool:
    """Save data to pickle file"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving pickle file {filepath}: {e}")
        return False

def load_csv_safely(filepath: Path, **kwargs) -> pd.DataFrame:
    """Load CSV file with error handling"""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"Error loading CSV file {filepath}: {e}")
        return pd.DataFrame()

def parse_list_string(text: str) -> List[str]:
    """Parse string representation of list safely"""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    try:
        # Handle different list formats
        if text.startswith('[') and text.endswith(']'):
            return ast.literal_eval(text)
        elif ',' in text:
            return [item.strip() for item in text.split(',')]
        else:
            return [text.strip()]
    except:
        return []

def calculate_health_score(recipe: Dict[str, Any]) -> float:
    """Calculate health score for a recipe"""
    score = 50  # Base score
    
    # Ingredients factor
    ingredients = recipe.get('ingredients', [])
    if ingredients:
        n_ingredients = len(ingredients)
        if n_ingredients >= 8:
            score += 20
        elif n_ingredients >= 5:
            score += 15
        elif n_ingredients >= 3:
            score += 10
    
    # Cooking time factor
    minutes = recipe.get('minutes', 30)
    if minutes < 30:
        score += 15
    elif minutes < 60:
        score += 10
    elif minutes < 90:
        score += 5
    
    # Tags factor
    tags = recipe.get('tags', [])
    healthy_tags = ['healthy', 'low-fat', 'low-carb', 'vegetarian', 'vegan', 'gluten-free', 'organic']
    for tag in tags:
        if any(healthy in tag.lower() for healthy in healthy_tags):
            score += 5
    
    # Dietary restrictions factor
    dietary_tags = ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'nut-free']
    for tag in tags:
        if any(dietary in tag.lower() for dietary in dietary_tags):
            score += 3
    
    return max(0, min(100, score))

def create_id_mappings(interactions_df: pd.DataFrame) -> Dict[str, Dict]:
    """Create ID mappings for users and recipes"""
    user_ids = interactions_df['user_id'].unique()
    recipe_ids = interactions_df['recipe_id'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}
    
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_recipe = {idx: recipe_id for recipe_id, idx in recipe_to_idx.items()}
    
    return {
        'user_to_idx': user_to_idx,
        'recipe_to_idx': recipe_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_recipe': idx_to_recipe
    }

def evaluate_recommendations(recommender, test_interactions: pd.DataFrame, 
                           recipes_df: pd.DataFrame, n_recommendations: int = 10) -> Dict[str, float]:
    """Evaluate recommendation model performance"""
    print("[DEBUG] Entered evaluate_recommendations")
    from sklearn.metrics import mean_squared_error
    
    predictions = []
    actuals = []
    
    # Get users that exist in both test set and training data
    test_users = test_interactions['user_id'].unique()
    print(f"[DEBUG] Number of test users: {len(test_users)}")
    valid_users = []
    
    for user_id in test_users:
        try:
            recommendations = recommender.recommend(user_id, n_recommendations)
            if recommendations:
                valid_users.append(user_id)
        except (ValueError, KeyError):
            continue
    print(f"[DEBUG] Number of valid users: {len(valid_users)}")
    
    if not valid_users:
        print("⚠️ No valid users found for evaluation")
        return {'rmse': float('inf'), 'mae': float('inf')}
    
    print(f"📊 Evaluating {len(valid_users)} valid users out of {len(test_users)} test users")
    
    for user_id in valid_users:
        user_test = test_interactions[test_interactions['user_id'] == user_id]
        try:
            recommendations = recommender.recommend(user_id, n_recommendations)
            recommended_recipe_ids = [rec[0] for rec in recommendations]
            for recipe_id in recommended_recipe_ids:
                if recipe_id in user_test['recipe_id'].values:
                    actual_rating = user_test[user_test['recipe_id'] == recipe_id]['rating'].iloc[0]
                    predicted_rating = min(5, max(1, recommendations[recommended_recipe_ids.index(recipe_id)][1] * 5))
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        except Exception as e:
            print(f"[DEBUG] Error for user {user_id}: {e}")
            continue
    print(f"[DEBUG] Number of predictions: {len(predictions)}")
    
    if not predictions:
        print("⚠️ No predictions generated for evaluation")
        return {'rmse': float('inf'), 'mae': float('inf')}
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
    print(f"[DEBUG] RMSE: {rmse}, MAE: {mae}")
    
    return {'rmse': rmse, 'mae': mae} 