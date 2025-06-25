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

def calculate_carbon_footprint(recipe: Dict[str, Any]) -> Tuple[float, List, List, List]:
    """Calculate carbon footprint based on ingredients"""
    carbon_factors = {
        # High impact
        'beef': 13.3, 'lamb': 39.2, 'pork': 7.2, 'chicken': 6.9, 'turkey': 6.9,
        'cheese': 13.5, 'milk': 3.0, 'eggs': 4.8, 'butter': 23.8,
        
        # Medium impact
        'fish': 3.0, 'shrimp': 12.0, 'salmon': 3.0, 'tuna': 3.0,
        'rice': 2.7, 'pasta': 1.4, 'bread': 1.4, 'potato': 0.2,
        
        # Low impact
        'tomato': 1.4, 'lettuce': 0.4, 'carrot': 0.4, 'onion': 0.4,
        'apple': 0.4, 'banana': 0.7, 'orange': 0.3, 'broccoli': 0.4,
        'spinach': 0.4, 'kale': 0.4, 'cucumber': 0.4, 'bell pepper': 0.4,
        'mushroom': 0.4, 'garlic': 0.4, 'ginger': 0.4, 'lemon': 0.3,
        'lime': 0.3, 'avocado': 0.4, 'olive oil': 6.0, 'vegetable oil': 3.8,
        
        # Very low impact
        'water': 0.0, 'salt': 0.0, 'pepper': 0.0, 'herbs': 0.0, 'spices': 0.0
    }
    
    ingredients = recipe.get('ingredients', [])
    total_carbon = 0
    high_impact_ingredients = []
    medium_impact_ingredients = []
    low_impact_ingredients = []
    
    for ingredient in ingredients:
        ingredient_lower = ingredient.lower()
        for key, factor in carbon_factors.items():
            if key in ingredient_lower:
                ingredient_carbon = factor * 0.1  # Assume 100g per ingredient
                total_carbon += ingredient_carbon
                
                if factor >= 5.0:
                    high_impact_ingredients.append((ingredient, factor))
                elif factor >= 1.0:
                    medium_impact_ingredients.append((ingredient, factor))
                else:
                    low_impact_ingredients.append((ingredient, factor))
                break
    
    return total_carbon, high_impact_ingredients, medium_impact_ingredients, low_impact_ingredients

def format_cooking_time(minutes: int) -> str:
    """Format cooking time in hours and minutes if over 60 minutes"""
    if minutes is None or pd.isna(minutes):
        return "N/A"
    
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes} min"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours}h"
        else:
            return f"{hours}h{remaining_minutes:02d}"

def get_user_level(points: int) -> str:
    """Calculate user level from points"""
    levels = [
        (0, "Novice Cook"),
        (100, "Home Chef"),
        (300, "Skilled Cook"),
        (600, "Master Chef"),
        (1000, "Culinary Expert")
    ]
    
    for threshold, title in reversed(levels):
        if points >= threshold:
            return title
    return "Novice Cook"

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
    from sklearn.metrics import mean_squared_error
    
    predictions = []
    actuals = []
    
    for user_id in test_interactions['user_id'].unique():
        user_test = test_interactions[test_interactions['user_id'] == user_id]
        
        # Get recommendations
        recommendations = recommender.recommend(user_id, n_recommendations)
        recommended_recipe_ids = [rec[0] for rec in recommendations]
        
        # Get actual ratings for recommended recipes
        for recipe_id in recommended_recipe_ids:
            if recipe_id in user_test['recipe_id'].values:
                actual_rating = user_test[user_test['recipe_id'] == recipe_id]['rating'].iloc[0]
                # Use recommendation score as prediction (normalize to 1-5 scale)
                predicted_rating = min(5, max(1, recommendations[recommended_recipe_ids.index(recipe_id)][1] * 5))
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
    
    if not predictions:
        return {'rmse': float('inf'), 'mae': float('inf')}
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
    
    return {'rmse': rmse, 'mae': mae}

def save_user_profile(profile: Dict, user_id: str, data_dir: str = "user_data") -> bool:
    """Save user profile to JSON"""
    try:
        filepath = Path(data_dir) / f"user_{user_id}.json"
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving user profile: {e}")
        return False

def load_user_profile(user_id: str, data_dir: str = "user_data") -> Dict:
    """Load user profile from JSON"""
    try:
        filepath = Path(data_dir) / f"user_{user_id}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading user profile: {e}")
    
    # Return default profile
    return {
        'user_id': user_id,
        'liked_recipes': [],
        'disliked_recipes': [],
        'rated_recipes': {},
        'points': 0,
        'level': 1,
        'achievements': []
    } 