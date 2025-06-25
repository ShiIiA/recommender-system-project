#!/usr/bin/env python3
"""
Check the current status of the recipe recommendation system
"""
import os
from pathlib import Path

def check_status():
    print("🔍 Checking project status...")
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        print(f"📁 Models directory exists with {len(list(models_dir.glob('*.pkl')))} files:")
        for file in models_dir.glob('*.pkl'):
            size = file.stat().st_size / (1024*1024)  # MB
            print(f"  - {file.name} ({size:.1f} MB)")
    else:
        print("❌ Models directory not found")
    
    # Check processed_data directory
    processed_dir = Path("processed_data")
    if processed_dir.exists():
        print(f"📁 Processed data directory exists with {len(list(processed_dir.glob('*')))} files:")
        for file in processed_dir.glob('*'):
            size = file.stat().st_size / (1024*1024)  # MB
            print(f"  - {file.name} ({size:.1f} MB)")
    else:
        print("❌ Processed data directory not found")
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        print(f"📁 Data directory exists with {len(csv_files)} CSV files:")
        for file in csv_files:
            size = file.stat().st_size / (1024*1024)  # MB
            print(f"  - {file.name} ({size:.1f} MB)")
    else:
        print("❌ Data directory not found")
    
    # Check if hybrid model exists
    hybrid_model = models_dir / "hybrid_recommender.pkl"
    if hybrid_model.exists():
        print("✅ Hybrid model found!")
    else:
        print("❌ Hybrid model not found")
    
    # Check if processed recipes exist
    processed_recipes = processed_dir / "recipes_processed.pkl"
    if processed_recipes.exists():
        print("✅ Processed recipes found!")
    else:
        print("❌ Processed recipes not found")

if __name__ == "__main__":
    check_status() 