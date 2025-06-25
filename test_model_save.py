#!/usr/bin/env python3
"""
Test script to verify model saving and loading
"""
import sys
import os
sys.path.append('src')

from recommendation_models import HybridRecommender
import pandas as pd
import pickle
from pathlib import Path

def test_model_save():
    """Test if we can create and save a simple hybrid model"""
    print("🧪 Testing model save/load functionality...")
    
    # Create a simple test dataset
    test_interactions = pd.DataFrame({
        'user_id': [0, 0, 1, 1, 2, 2],
        'recipe_id': [0, 1, 0, 2, 1, 2],
        'rating': [5, 4, 3, 5, 4, 3]
    })
    
    test_recipes = pd.DataFrame({
        'id': [0, 1, 2],
        'name': ['Recipe A', 'Recipe B', 'Recipe C'],
        'ingredients': [['tomato', 'cheese'], ['pasta', 'sauce'], ['chicken', 'rice']],
        'tags': [['italian', 'quick'], ['pasta', 'dinner'], ['asian', 'healthy']],
        'minutes': [30, 45, 60]
    })
    
    try:
        # Create and train model
        print("📝 Creating hybrid model...")
        model = HybridRecommender(collaborative_weight=0.7, content_weight=0.3)
        model.fit(test_interactions, test_recipes, n_components=2)
        print("✅ Model created successfully!")
        
        # Test saving
        print("💾 Testing model save...")
        model.save_model("models/test_hybrid.pkl")
        print("✅ Model saved successfully!")
        
        # Test loading
        print("📂 Testing model load...")
        loaded_model = HybridRecommender.load_model("models/test_hybrid.pkl")
        print("✅ Model loaded successfully!")
        
        # Test recommendation
        print("🔮 Testing recommendation...")
        recommendations = loaded_model.recommend(0, n_recommendations=2)
        print(f"✅ Recommendations generated: {recommendations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_save()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!") 