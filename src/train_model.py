"""
Training script for the recipe recommendation system
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from src.recommendation_models import HybridRecommender, ContentBasedRecommender, CollaborativeRecommender
from src.utils.utils import load_csv_safely, parse_list_string, calculate_health_score, create_id_mappings, evaluate_recommendations

def load_and_preprocess_data():
    """Load and preprocess the recipe and interaction data"""
    print("📊 Loading and preprocessing data...")
    
    # Load recipes
    recipes_path = Path("data/RAW_recipes.csv")
    if not recipes_path.exists():
        print("❌ Recipe data not found. Please ensure RAW_recipes.csv is in the data/ directory.")
        return None, None
    
    recipes_df = load_csv_safely(recipes_path)
    if recipes_df.empty:
        print("❌ Failed to load recipe data.")
        return None, None
    
    # Load interactions
    interactions_path = Path("data/RAW_interactions.csv")
    if not interactions_path.exists():
        print("❌ Interaction data not found. Please ensure RAW_interactions.csv is in the data/ directory.")
        return None, None
    
    interactions_df = load_csv_safely(interactions_path)
    if interactions_df.empty:
        print("❌ Failed to load interaction data.")
        return None, None
    
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
    
    # Filter to only include recipes that have interactions
    recipe_ids_with_interactions = interactions_df['recipe_id'].unique()
    recipes_df = recipes_df[recipes_df['id'].isin(recipe_ids_with_interactions)]
    
    print(f"✅ Loaded {len(recipes_df)} recipes and {len(interactions_df)} interactions")
    return recipes_df, interactions_df

def train_models(recipes_df, interactions_df):
    """Train all recommendation models"""
    print("\n🤖 Training recommendation models...")
    
    # Create output directory
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # Split interactions for evaluation
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42, stratify=interactions_df['rating']
    )
    
    models = {}
    results = {}
    
    # 1. Train Hybrid Model
    print("\n🔄 Training Hybrid Model...")
    hybrid_model = HybridRecommender(collaborative_weight=0.7, content_weight=0.3)
    hybrid_model.fit(train_interactions, recipes_df, n_components=100)
    
    # Evaluate hybrid model
    hybrid_results = evaluate_recommendations(hybrid_model, test_interactions, recipes_df)
    results['hybrid'] = hybrid_results
    
    # Save hybrid model
    hybrid_model.save_model(output_dir / "hybrid_recommender.pkl")
    models['hybrid'] = hybrid_model
    
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
    collaborative_model = CollaborativeRecommender(n_components=100)
    collaborative_model.fit(train_interactions)
    
    # Evaluate collaborative model
    collab_results = evaluate_recommendations(collaborative_model, test_interactions, recipes_df)
    results['collaborative'] = collab_results
    
    # Save collaborative model
    with open(output_dir / "collaborative_model.pkl", 'wb') as f:
        pickle.dump(collaborative_model, f)
    models['collaborative'] = collaborative_model
    
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