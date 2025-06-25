#!/usr/bin/env python3
"""
Create missing files for the recipe recommendation app
"""
import sys
import os
sys.path.append('src')

import pandas as pd
import pickle
from pathlib import Path
import ast

def create_missing_files():
    """Create the missing files that the app needs"""
    print("🔧 Creating missing files for the app...")
    
    # Load raw data
    print("📊 Loading raw data...")
    recipes_df = pd.read_csv("data/RAW_recipes.csv")
    interactions_df = pd.read_csv("data/RAW_interactions.csv")
    
    # Parse list columns
    print("🔧 Processing recipe data...")
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
    recipes_df['tags'] = recipes_df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
    recipes_df['steps'] = recipes_df['steps'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
    
    # Calculate health scores
    def calculate_health_score(recipe):
        score = 50  # Base score
        ingredients = recipe.get('ingredients', [])
        if ingredients:
            n_ingredients = len(ingredients)
            if n_ingredients >= 8:
                score += 20
            elif n_ingredients >= 5:
                score += 15
            elif n_ingredients >= 3:
                score += 10
        
        minutes = recipe.get('minutes', 30)
        if minutes < 30:
            score += 15
        elif minutes < 60:
            score += 10
        elif minutes < 90:
            score += 5
        
        tags = recipe.get('tags', [])
        healthy_tags = ['healthy', 'low-fat', 'low-carb', 'vegetarian', 'vegan', 'gluten-free', 'organic']
        for tag in tags:
            if any(healthy in tag.lower() for healthy in healthy_tags):
                score += 5
        
        dietary_tags = ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'nut-free']
        for tag in tags:
            if any(dietary in tag.lower() for dietary in dietary_tags):
                score += 3
        
        return max(0, min(100, score))
    
    recipes_df['health_score'] = recipes_df.apply(calculate_health_score, axis=1)
    recipes_df['n_ingredients'] = recipes_df['ingredients'].apply(len)
    
    # Create processed_data directory
    processed_dir = Path("processed_data")
    processed_dir.mkdir(exist_ok=True)
    
    # Save processed recipes
    print("💾 Saving processed recipes...")
    recipes_df.to_pickle(processed_dir / "recipes_processed.pkl")
    recipes_df.to_pickle(processed_dir / "recipes_full.pkl")
    
    # Save processed interactions
    print("💾 Saving processed interactions...")
    interactions_df.to_pickle(processed_dir / "interactions_processed.pkl")
    interactions_df.to_pickle(processed_dir / "interactions_full.pkl")
    
    # Create simple ID mappings
    print("🔗 Creating ID mappings...")
    user_ids = interactions_df['user_id'].unique()
    recipe_ids = interactions_df['recipe_id'].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}
    
    id_mappings = {
        'user_to_idx': user_to_idx,
        'idx_to_user': {v: k for k, v in user_to_idx.items()},
        'recipe_to_idx': recipe_to_idx,
        'idx_to_recipe': {v: k for k, v in recipe_to_idx.items()}
    }
    
    with open(processed_dir / "id_mappings.pkl", 'wb') as f:
        pickle.dump(id_mappings, f)
    
    # Create a simple hybrid model placeholder
    print("🤖 Creating simple hybrid model...")
    from recommendation_models import HybridRecommender
    
    # Sample a small dataset for the model
    sample_users = user_ids[:1000]
    sample_recipes = recipe_ids[:5000]
    
    sample_interactions = interactions_df[
        interactions_df['user_id'].isin(sample_users) & 
        interactions_df['recipe_id'].isin(sample_recipes)
    ]
    sample_recipes_df = recipes_df[recipes_df['id'].isin(sample_recipes)]
    
    # Remap IDs to contiguous integers
    user_id_map = {id_: idx for idx, id_ in enumerate(sample_interactions['user_id'].unique())}
    recipe_id_map = {id_: idx for idx, id_ in enumerate(sample_interactions['recipe_id'].unique())}
    
    sample_interactions['user_id'] = sample_interactions['user_id'].map(user_id_map)
    sample_interactions['recipe_id'] = sample_interactions['recipe_id'].map(recipe_id_map)
    sample_recipes_df['id'] = sample_recipes_df['id'].map(recipe_id_map)
    
    # Create and save hybrid model
    hybrid_model = HybridRecommender(collaborative_weight=0.7, content_weight=0.3)
    hybrid_model.fit(sample_interactions, sample_recipes_df, n_components=20)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    hybrid_model.save_model(models_dir / "hybrid_recommender.pkl")
    
    print("✅ All missing files created successfully!")
    print(f"📊 Processed {len(recipes_df)} recipes and {len(interactions_df)} interactions")
    print(f"🤖 Created hybrid model with {len(sample_users)} users and {len(sample_recipes)} recipes")

if __name__ == "__main__":
    create_missing_files() 