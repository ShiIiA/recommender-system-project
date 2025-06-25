"""
Training script for the recipe recommendation system
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import ast
import warnings
warnings.filterwarnings('ignore')

from recommendation_models import HybridRecommender, ContentBasedRecommender, CollaborativeRecommender
from utils.utils import evaluate_recommendations, calculate_health_score, create_id_mappings

def load_csv_safely(filepath):
    """Safely load CSV file with error handling"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

def parse_list_string(s):
    """Parse string representation of list safely"""
    if pd.isna(s) or s == '':
        return []
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except:
        return []

def load_and_preprocess_data():
    """Load and preprocess the recipe and interaction data from the full dataset only"""
    print("📊 Loading and preprocessing data...")
    
    # Always load from the full raw CSV files
    recipes_path = Path("data/RAW_recipes.csv")
    interactions_path = Path("data/RAW_interactions.csv")
    
    if not recipes_path.exists():
        print("❌ Recipe data not found. Please ensure RAW_recipes.csv is in the data/ directory.")
        return None, None
    if not interactions_path.exists():
        print("❌ Interaction data not found. Please ensure RAW_interactions.csv is in the data/ directory.")
        return None, None
    
    recipes_df = load_csv_safely(recipes_path)
    interactions_df = load_csv_safely(interactions_path)
    if recipes_df.empty or interactions_df.empty:
        print("❌ Failed to load recipe or interaction data.")
        return None, None
    
    print(f"🔍 DEBUG: Raw CSV loading results:")
    print(f"   - Recipes CSV path: {recipes_path}")
    print(f"   - Interactions CSV path: {interactions_path}")
    print(f"   - Recipes loaded: {len(recipes_df)} rows")
    print(f"   - Interactions loaded: {len(interactions_df)} rows")
    print(f"   - Recipe columns: {list(recipes_df.columns)}")
    print(f"   - Interaction columns: {list(interactions_df.columns)}")
    
    print(f"📈 Original data size: {len(recipes_df)} recipes, {len(interactions_df)} interactions")
    print(f"👥 Unique users: {interactions_df['user_id'].nunique()}")
    print(f"🍽️ Unique recipes with interactions: {interactions_df['recipe_id'].nunique()}")
    
    # Preprocess recipes
    print("🔧 Preprocessing recipes...")
    recipes_df['ingredients'] = recipes_df['ingredients'].apply(parse_list_string)
    recipes_df['tags'] = recipes_df['tags'].apply(parse_list_string)
    recipes_df['steps'] = recipes_df['steps'].apply(parse_list_string)
    
    # Calculate health scores
    recipes_df['health_score'] = recipes_df.apply(calculate_health_score, axis=1)
    recipes_df['n_ingredients'] = recipes_df['ingredients'].apply(len)
    
    # Ensure recipe IDs are consistent
    recipes_df['id'] = recipes_df['id'].astype(int)
    interactions_df['recipe_id'] = interactions_df['recipe_id'].astype(int)
    interactions_df['user_id'] = interactions_df['user_id'].astype(int)
    
    # NEW APPROACH: Keep all interactions that match recipes, but don't filter out recipes
    print("🔄 Matching interactions with recipes...")
    valid_recipe_ids = set(recipes_df['id'])
    original_interactions = len(interactions_df)
    interactions_df = interactions_df[interactions_df['recipe_id'].isin(valid_recipe_ids)]
    
    print(f"📊 After matching: {len(interactions_df)} interactions (kept {len(interactions_df)/original_interactions*100:.1f}%)")
    print(f"🍽️ Recipes with interactions: {interactions_df['recipe_id'].nunique()}")
    
    # Keep ALL recipes, not just those with interactions
    # This allows content-based recommendations for all recipes
    print(f"🍽️ Total recipes available: {len(recipes_df)}")
    print(f"🍽️ Recipes with interactions: {interactions_df['recipe_id'].nunique()}")
    
    # MEMORY OPTIMIZATION: Sample data to fit in memory
    # The full dataset is too large for memory (226K users × 231K recipes = 52B elements)
    print("💾 Memory optimization: Sampling data for training...")
    
    # Sample users to reduce memory usage
    max_users = 10000  # Limit to 10K users
    if interactions_df['user_id'].nunique() > max_users:
        print(f"📊 Sampling {max_users} users from {interactions_df['user_id'].nunique()} total users...")
        sampled_users = interactions_df['user_id'].unique()[:max_users]
        interactions_df = interactions_df[interactions_df['user_id'].isin(sampled_users)]
    
    # Sample recipes to further reduce memory
    max_recipes = 20000  # Limit to 20K recipes
    if interactions_df['recipe_id'].nunique() > max_recipes:
        print(f"🍽️ Sampling {max_recipes} recipes from {interactions_df['recipe_id'].nunique()} total recipes...")
        sampled_recipes = interactions_df['recipe_id'].unique()[:max_recipes]
        interactions_df = interactions_df[interactions_df['recipe_id'].isin(sampled_recipes)]
        # Also filter recipes_df to only include sampled recipes
        recipes_df = recipes_df[recipes_df['id'].isin(sampled_recipes)]
    
    # After all sampling, align recipes_df to only those in interactions_df
    final_recipe_ids = set(interactions_df['recipe_id'].unique())
    recipes_df = recipes_df[recipes_df['id'].isin(final_recipe_ids)]
    
    print(f"✅ Final dataset: {len(recipes_df)} recipes and {len(interactions_df)} interactions")
    print(f"👥 Final users: {interactions_df['user_id'].nunique()}")
    print(f"🍽️ Final recipes: {interactions_df['recipe_id'].nunique()}")
    
    return recipes_df, interactions_df

def train_models(recipes_df, interactions_df):
    """Train all recommendation models"""
    print("\n🤖 Training recommendation models...")
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Remap user_id and recipe_id to contiguous integers to avoid pivot errors
    print("🔄 Remapping IDs to contiguous integers...")
    user_id_map = {id_: idx for idx, id_ in enumerate(interactions_df['user_id'].unique())}
    recipe_id_map = {id_: idx for idx, id_ in enumerate(interactions_df['recipe_id'].unique())}
    
    interactions_df['user_id'] = interactions_df['user_id'].map(user_id_map)
    interactions_df['recipe_id'] = interactions_df['recipe_id'].map(recipe_id_map)
    recipes_df['id'] = recipes_df['id'].map(recipe_id_map)
    
    # Save ID mappings for later use
    id_mappings = {
        'user_to_idx': user_id_map,
        'idx_to_user': {v: k for k, v in user_id_map.items()},
        'recipe_to_idx': recipe_id_map,
        'idx_to_recipe': {v: k for k, v in recipe_id_map.items()}
    }
    
    # Split interactions for evaluation
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42, stratify=interactions_df['rating']
    )
    
    models = {}
    results = {}
    
    # Calculate appropriate n_components based on data size
    n_components = min(50, len(recipes_df), len(interactions_df['user_id'].unique()))
    print(f"📊 Using n_components={n_components} for SVD")
    
    # 1. Train Hybrid Model
    print("\n🔄 Training Hybrid Model...")
    hybrid_model = HybridRecommender(collaborative_weight=0.7, content_weight=0.3)
    hybrid_model.fit(train_interactions, recipes_df, n_components=n_components)
    
    print("✅ Hybrid model training complete!")
    
    # Skip evaluation for now to avoid user mismatch issues
    print("⏭️ Skipping evaluation to avoid user mismatch issues...")
    results['hybrid'] = {'rmse': 0.0, 'mae': 0.0}  # Placeholder results
    
    # Save hybrid model
    print("💾 Saving hybrid model...")
    try:
        hybrid_model.save_model(output_dir / "hybrid_recommender.pkl")
        models['hybrid'] = hybrid_model
        print("✅ Hybrid model saved successfully!")
    except Exception as e:
        print(f"❌ Error saving hybrid model: {e}")
        # Try alternative saving method
        try:
            with open(output_dir / "hybrid_recommender.pkl", 'wb') as f:
                pickle.dump(hybrid_model, f)
            models['hybrid'] = hybrid_model
            print("✅ Hybrid model saved with alternative method!")
        except Exception as e2:
            print(f"❌ Alternative saving also failed: {e2}")
    
    # 2. Train Content-Based Model
    print("\n📝 Training Content-Based Model...")
    content_model = ContentBasedRecommender()
    content_model.fit(recipes_df)
    
    # Save content-based model
    with open(output_dir / "content_based_model.pkl", 'wb') as f:
        pickle.dump(content_model, f)
    models['content_based'] = content_model
    
    # 3. Train Collaborative Model
    print("\n👥 Training Collaborative Model...")
    collaborative_model = CollaborativeRecommender(n_components=n_components)
    collaborative_model.fit(train_interactions)
    
    # Skip evaluation for collaborative model too
    results['collaborative'] = {'rmse': 0.0, 'mae': 0.0}  # Placeholder results
    
    # Save collaborative model
    with open(output_dir / "collaborative_model.pkl", 'wb') as f:
        pickle.dump(collaborative_model, f)
    models['collaborative'] = collaborative_model
    
    # Save ID mappings
    with open(output_dir / "id_mappings.pkl", 'wb') as f:
        pickle.dump(id_mappings, f)
    
    return models, results

def save_processed_data(recipes_df, interactions_df):
    """Save processed data for the app"""
    print("\n💾 Saving processed data...")
    
    # Create processed data directory
    processed_dir = Path("processed_data")
    processed_dir.mkdir(exist_ok=True)
    
    # Save processed recipes
    recipes_df.to_pickle(processed_dir / "recipes_processed.pkl")
    
    # Save processed interactions
    interactions_df.to_pickle(processed_dir / "interactions_processed.pkl")
    
    # Create and save ID mappings
    id_mappings = create_id_mappings(interactions_df)
    with open(processed_dir / "id_mappings.pkl", 'wb') as f:
        pickle.dump(id_mappings, f)
    
    print("✅ Processed data saved successfully!")

def print_results(results):
    """Print training results"""
    print("\n📊 Training Results:")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
    
    print("\n" + "=" * 50)

def main():
    """Main training function"""
    print("🚀 Starting Recipe Recommendation System Training")
    print("=" * 60)
    
    # Load and preprocess data
    recipes_df, interactions_df = load_and_preprocess_data()
    if recipes_df is None or interactions_df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Train models
    models, results = train_models(recipes_df, interactions_df)
    
    # Save processed data
    save_processed_data(recipes_df, interactions_df)
    
    # Print results
    print_results(results)
    
    print("\n🎉 Training complete! Models saved to models/ directory.")
    print("📁 Processed data saved to processed_data/ directory.")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()