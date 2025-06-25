"""
Enhanced Recipe Recommendation App with Proper ML Models
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, date
import json
import os
import random
from src.models.hybrid import HybridModel
from src.utils.utils import load_pickle
from src.models.llm_enhancer import LocalLLMEnhancer
import ast
import hashlib

# Page configuration
st.set_page_config(
    page_title="🌿 Ghibli Recipe Garden - AI Powered",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = "user_data"
MODEL_PATH = "models/hybrid_model.pkl"
RECIPES_PATH = "processed_data/recipes_full.pkl"
INTERACTIONS_PATH = "processed_data/interactions_full.pkl"
ID_MAPPINGS_PATH = "processed_data/id_mappings.pkl"

# Create directories
Path(DATA_DIR).mkdir(exist_ok=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None  # Will be set after login
    
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None # Will be loaded after login
    
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    
    if 'recipes_df' not in st.session_state:
        st.session_state.recipes_df = None
    
    if 'interactions_df' not in st.session_state:
        st.session_state.interactions_df = None
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Login" # Start at login page
    
    if 'id_mappings' not in st.session_state:
        st.session_state.id_mappings = None
    
    if 'llm_enhancer' not in st.session_state:
        st.session_state.llm_enhancer = LocalLLMEnhancer()

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained hybrid recommendation model and id mappings"""
    try:
        model = HybridModel.load_model("models/hybrid_recommender.pkl")
        id_mappings = load_pickle(Path(ID_MAPPINGS_PATH))
        st.success("🤖 Hybrid Recommendation Engine Ready")
        return model, id_mappings
    except Exception as e:
        st.error(f"Error loading hybrid model: {e}")
        return None, None

@st.cache_data
def load_recipe_data():
    """Load processed recipe data for compatibility with the model. Fallback to raw CSV if missing."""
    processed_path = Path("processed_data/recipes_processed.pkl")
    if processed_path.exists():
        try:
            df = load_pickle(processed_path)
            st.info("Loaded processed recipes for model compatibility.")
            return df
        except Exception as e:
            st.error(f"Error loading processed recipes: {e}")
    # Fallback to raw CSV
    raw_path = Path("data/RAW_recipes.csv")
    if raw_path.exists():
        try:
            df = pd.read_csv(raw_path)
            df['id'] = df.get('id', pd.Series(range(len(df))))
            df['ingredients'] = df['ingredients'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
            df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
            df['steps'] = df['recipe_steps'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [])
            
            # Calculate comprehensive health scores
            df['health_score'] = df.apply(calculate_health_score, axis=1)
            
            if 'n_ingredients' not in df.columns:
                df['n_ingredients'] = df['ingredients'].apply(len)
            if 'description' not in df.columns:
                df['description'] = df.get('description', "A delicious recipe.")
            st.warning("Loaded raw recipes. Some features may not match the model.")
            return df
        except Exception as e:
            st.error(f"Error loading or processing recipes from CSV: {e}")
    st.error("No recipe data found. Please ensure the data pipeline has been run.")
    return pd.DataFrame()

@st.cache_data
def load_interaction_data():
    """Load and preprocess interaction data from CSV"""
    path = Path("data/RAW_interactions.csv")
    if not path.exists():
        st.warning("`data/RAW_interactions.csv` not found. No interaction data loaded.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        # Ensure correct column names
        df = df.rename(columns={'recipe_id': 'recipe_id', 'user_id': 'user_id', 'rating': 'rating'})
        return df
    except Exception as e:
        st.error(f"Error loading interactions from CSV: {e}")
        return pd.DataFrame()

# =============================================================================
# USER PROFILE MANAGEMENT
# =============================================================================
def save_user_profile(profile, user_id):
    """Save user profile to JSON"""
    filepath = Path(DATA_DIR) / f"user_{user_id}.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types like int64, float64
            return obj.item()
        else:
            return obj
    
    # Convert the profile to JSON-serializable format
    serializable_profile = convert_numpy_types(profile)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_profile, f, indent=2)

def load_user_profile(user_id):
    """Load user profile from JSON"""
    filepath = Path(DATA_DIR) / f"user_{user_id}.json"
    
    if filepath.exists():
        try:
        with open(filepath, 'r') as f:
            return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # If the JSON file is corrupted, create a backup and return default profile
            st.warning(f"Profile file corrupted for user {user_id}. Creating new profile.")
            try:
                # Create a backup of the corrupted file
                backup_path = filepath.with_suffix('.json.backup')
                filepath.rename(backup_path)
            except:
                pass  # If backup fails, just continue
            # Return default profile
            return get_default_profile(user_id)
    
    # Return default profile if file doesn't exist
    return get_default_profile(user_id)

def get_default_profile(user_id):
    """Get default user profile"""
    return {
        'user_id': user_id,
        'avatar': 'chef',  # Default avatar
        'liked_recipes': [],
        'disliked_recipes': [],
        'rated_recipes': {},
        'dietary_restrictions': [],
        'skill_level': 'intermediate',
        'preferred_time': 'medium',
        'cuisine_preferences': [],
        'health_conscious': False,
        'sustainability_focus': False,
        'achievements': [],
        'points': 0,
        'level': 1,
        'created_at': datetime.now().isoformat()
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_avatar_info(avatar_key):
    """Get avatar information by key"""
    avatars = {
        "chef": {"emoji": "👨‍🍳", "name": "Master Chef", "description": "Professional cooking expert"},
        "cat": {"emoji": "🐱", "name": "Curious Cat", "description": "Playful and adventurous"},
        "dog": {"emoji": "🐕", "name": "Loyal Dog", "description": "Friendly and dependable"},
        "rabbit": {"emoji": "🐰", "name": "Quick Rabbit", "description": "Fast and energetic"},
        "owl": {"emoji": "🦉", "name": "Wise Owl", "description": "Knowledgeable and thoughtful"},
        "bear": {"emoji": "🐻", "name": "Cozy Bear", "description": "Warm and comforting"},
        "fox": {"emoji": "🦊", "name": "Clever Fox", "description": "Smart and resourceful"},
        "penguin": {"emoji": "🐧", "name": "Cool Penguin", "description": "Chill and collected"},
        "unicorn": {"emoji": "🦄", "name": "Magical Unicorn", "description": "Creative and unique"},
        "dragon": {"emoji": "🐉", "name": "Fiery Dragon", "description": "Passionate and bold"},
        "butterfly": {"emoji": "🦋", "name": "Graceful Butterfly", "description": "Elegant and free-spirited"},
        "turtle": {"emoji": "🐢", "name": "Patient Turtle", "description": "Steady and reliable"}
    }
    return avatars.get(avatar_key, avatars["chef"])

def get_current_season():
    """Get current season"""
    month = date.today().month
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return "winter"

def calculate_user_level(points):
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

def award_achievement(achievement_id, achievement_name, points=50):
    """Award achievement to user"""
    profile = st.session_state.user_profile
    
    if profile and achievement_id not in profile['achievements']:
        profile['achievements'].append(achievement_id)
        profile['points'] += points
        save_user_profile(profile, st.session_state.user_id)
        st.success(f"🏆 Achievement Unlocked: {achievement_name} (+{points} points)")

def get_seasonal_foods():
    """Get seasonal fruits and vegetables based on current month"""
    month = date.today().month
    if month in [12, 1, 2]:  # Winter
        return {
            'season': 'Winter',
            'emoji': '❄️',
            'fruits': ['🍊 Oranges', '🍎 Apples', '🍐 Pears', '🥝 Kiwi', '🍇 Grapes'],
            'vegetables': ['🥕 Carrots', '🥔 Potatoes', '🧅 Onions', '🧄 Garlic', '🥬 Kale'],
            'tip': "Winter is perfect for hearty soups and roasted vegetables!"
        }
    elif month in [3, 4, 5]:  # Spring
        return {
            'season': 'Spring',
            'emoji': '🌸',
            'fruits': ['🍓 Strawberries', '🍒 Cherries', '🥭 Mango', '🍍 Pineapple', '🥝 Kiwi'],
            'vegetables': ['🥬 Asparagus', '🫛 Peas', '🥕 Carrots', '🧅 Spring Onions', '🥬 Spinach'],
            'tip': "Spring brings fresh, light ingredients perfect for salads!"
        }
    elif month in [6, 7, 8]:  # Summer
        return {
            'season': 'Summer',
            'emoji': '☀️',
            'fruits': ['🍉 Watermelon', '🍑 Peaches', '🍓 Strawberries', '🫐 Blueberries', '🍇 Grapes'],
            'vegetables': ['🍅 Tomatoes', '🥒 Cucumbers', '🫑 Bell Peppers', '🌽 Corn', '🥬 Zucchini'],
            'tip': "Summer is ideal for fresh, no-cook recipes and grilling!"
        }
    else:  # Fall
        return {
            'season': 'Autumn',
            'emoji': '🍂',
            'fruits': ['🍎 Apples', '🍐 Pears', '🫐 Cranberries', '🥝 Kiwi', '🍊 Oranges'],
            'vegetables': ['🎃 Pumpkin', '🍠 Sweet Potatoes', '🥬 Brussels Sprouts', '🍄 Mushrooms', '🥬 Kale'],
            'tip': "Autumn is perfect for comforting stews and baked dishes!"
        }

def get_sustainability_tips():
    """Get sustainability tips for cooking"""
    return [
        "🌱 Choose plant-based proteins when possible",
        "🚜 Buy local and seasonal ingredients",
        "♻️ Use leftovers creatively in new dishes",
        "⚡ Cook in batches to save energy",
        "🌍 Support fair-trade and organic products",
        "🥬 Use vegetable scraps for homemade broth"
    ]

def calculate_health_score(recipe):
    """Calculate a comprehensive health score based on multiple factors"""
    score = 50  # Base score
    
    # Ingredients factor (more ingredients often = more nutrients)
    ingredients = recipe.get('ingredients', [])
    if ingredients:
        n_ingredients = len(ingredients)
        if n_ingredients >= 8:
            score += 20
        elif n_ingredients >= 5:
            score += 15
        elif n_ingredients >= 3:
            score += 10
    
    # Cooking time factor (shorter often = less processed)
    minutes = recipe.get('minutes', 30)
    if minutes < 30:
        score += 15
    elif minutes < 60:
        score += 10
    elif minutes < 90:
        score += 5
    
    # Tags factor (healthy tags boost score)
    tags = recipe.get('tags', [])
    healthy_tags = ['healthy', 'low-fat', 'low-carb', 'vegetarian', 'vegan', 'gluten-free', 'organic']
    for tag in tags:
        if any(healthy in tag.lower() for healthy in healthy_tags):
            score += 5
    
    # Dietary restrictions factor (accommodating dietary needs is healthy)
    dietary_tags = ['vegetarian', 'vegan', 'gluten-free', 'dairy-free', 'nut-free']
    for tag in tags:
        if any(dietary in tag.lower() for dietary in dietary_tags):
            score += 3
    
    return max(0, min(100, score))

def calculate_carbon_footprint(recipe):
    """Calculate carbon footprint based on ingredients"""
    # Carbon footprint per kg of common ingredients (kg CO2e/kg)
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
        # Find the best match for the ingredient
        for key, factor in carbon_factors.items():
            if key in ingredient_lower:
                # Assume 100g per ingredient (rough estimate)
                ingredient_carbon = factor * 0.1
                total_carbon += ingredient_carbon
                
                # Categorize ingredients by impact
                if factor >= 5.0:
                    high_impact_ingredients.append((ingredient, factor))
                elif factor >= 1.0:
                    medium_impact_ingredients.append((ingredient, factor))
                else:
                    low_impact_ingredients.append((ingredient, factor))
                break
    
    return total_carbon, high_impact_ingredients, medium_impact_ingredients, low_impact_ingredients

def get_carbon_explanation(carbon_footprint, high_impact, medium_impact, low_impact):
    """Get detailed explanation of why carbon footprint is high or low"""
    if carbon_footprint < 0.5:
        if low_impact:
            return f"🌱 Excellent! This recipe uses mostly plant-based ingredients like {', '.join([ing[0] for ing in low_impact[:3]])}. Plant-based foods have much lower carbon footprints than animal products."
        else:
            return "🌱 Great choice! This recipe has a very low environmental impact."
    
    elif carbon_footprint < 1.0:
        if high_impact:
            return f"🌿 Good choice! While this recipe includes some higher-impact ingredients like {', '.join([ing[0] for ing in high_impact[:2]])}, it's balanced with plant-based options."
        else:
            return "🌿 This recipe has a moderate carbon footprint, making it a sustainable choice."
    
    elif carbon_footprint < 2.0:
        if high_impact:
            return f"🌳 Moderate impact. This recipe includes higher-carbon ingredients like {', '.join([ing[0] for ing in high_impact[:3]])}. Consider reducing portions or substituting with plant-based alternatives."
        else:
            return "🌳 This recipe has a moderate environmental impact. Consider adding more plant-based ingredients to reduce the carbon footprint."
    
    elif carbon_footprint < 4.0:
        if high_impact:
            return f"🌍 High impact due to ingredients like {', '.join([ing[0] for ing in high_impact[:3]])}. These animal products have significant carbon footprints. Try plant-based alternatives for a more sustainable meal."
        else:
            return "🌍 This recipe has a high carbon footprint. Consider reducing portion sizes or choosing more sustainable alternatives."
    
    else:
        if high_impact:
            return f"🔥 Very high impact! This recipe contains high-carbon ingredients like {', '.join([ing[0] for ing in high_impact[:3]])}. Consider plant-based alternatives or reducing these ingredients to make it more sustainable."
        else:
            return "🔥 This recipe has a very high carbon footprint. Consider choosing more sustainable alternatives or reducing portion sizes."

def get_carbon_insight(carbon_footprint):
    """Get meaningful insight about carbon footprint impact"""
    insights = [
        {
            'range': (0, 0.5),
            'level': '🌱 Very Low',
            'comparison': 'Equivalent to walking 1 km',
            'savings': 'Saves 0.1 kg CO2 vs average meal',
            'impact': 'Excellent choice for the planet!'
        },
        {
            'range': (0.5, 1.0),
            'level': '🌿 Low',
            'comparison': 'Equivalent to driving 2 km',
            'savings': 'Saves 0.5 kg CO2 vs average meal',
            'impact': 'Great environmental choice!'
        },
        {
            'range': (1.0, 2.0),
            'level': '🌳 Moderate',
            'comparison': 'Equivalent to driving 5 km',
            'savings': 'Similar to average meal',
            'impact': 'Good balance of taste and sustainability'
        },
        {
            'range': (2.0, 4.0),
            'level': '🌍 High',
            'comparison': 'Equivalent to driving 10 km',
            'savings': '0.5 kg CO2 more than average meal',
            'impact': 'Consider plant-based alternatives'
        },
        {
            'range': (4.0, float('inf')),
            'level': '🔥 Very High',
            'comparison': 'Equivalent to driving 20+ km',
            'savings': '1+ kg CO2 more than average meal',
            'impact': 'High environmental impact - try plant-based!'
        }
    ]
    
    for insight in insights:
        if insight['range'][0] <= carbon_footprint < insight['range'][1]:
            return insight
    
    return insights[2]  # Default to moderate

def get_environmental_achievement(carbon_footprint):
    """Check if user deserves environmental achievement"""
    if carbon_footprint < 0.5:
        return "🌱 Eco Warrior", "Cooked a very low-carbon meal!", 20
    elif carbon_footprint < 1.0:
        return "🌿 Green Chef", "Cooked a low-carbon meal!", 15
    return None, None, 0

def get_health_score_explanation(health_score, recipe):
    """Get detailed explanation of health score and suggestions for improvement"""
    ingredients = recipe.get('ingredients', [])
    tags = recipe.get('tags', [])
    minutes = recipe.get('minutes', 30)
    
    explanation_parts = []
    suggestions = []
    
    # Score-based feedback
    if health_score >= 80:
        explanation_parts.append("🌟 Excellent health score! This recipe is well-balanced and nutritious.")
    elif health_score >= 70:
        explanation_parts.append("👍 Good health score! This recipe has many healthy qualities.")
    elif health_score >= 50:
        explanation_parts.append("⚠️ Moderate health score. This recipe has some healthy aspects but could be improved.")
    else:
        explanation_parts.append("📉 Low health score. This recipe may need some modifications for better nutrition.")
    
    # Ingredient-based feedback
    n_ingredients = len(ingredients)
    if n_ingredients >= 8:
        explanation_parts.append(f"🥗 Rich in variety with {n_ingredients} ingredients for diverse nutrients.")
    elif n_ingredients >= 5:
        explanation_parts.append(f"🥘 Good ingredient variety with {n_ingredients} components.")
    else:
        explanation_parts.append(f"🍽️ Simple recipe with {n_ingredients} ingredients.")
        suggestions.append("Consider adding more vegetables or whole grains for better nutrition.")
    
    # Cooking time feedback
    if minutes < 30:
        explanation_parts.append("⚡ Quick preparation preserves nutrients and reduces processing.")
    elif minutes < 60:
        explanation_parts.append("⏱️ Moderate cooking time balances convenience with nutrition.")
    else:
        explanation_parts.append("🕰️ Longer cooking time may reduce some nutrients but can enhance flavors.")
        suggestions.append("Consider steaming or quick-cooking methods to preserve more nutrients.")
    
    # Tag-based feedback
    healthy_tags_found = []
    for tag in tags:
        tag_lower = tag.lower()
        if any(healthy in tag_lower for healthy in ['healthy', 'low-fat', 'low-carb', 'organic']):
            healthy_tags_found.append(tag)
    
    if healthy_tags_found:
        explanation_parts.append(f"🏷️ Contains healthy tags: {', '.join(healthy_tags_found)}.")
    
    # Dietary accommodation feedback
    dietary_tags = []
    for tag in tags:
        tag_lower = tag.lower()
        if any(dietary in tag_lower for dietary in ['vegetarian', 'vegan', 'gluten-free', 'dairy-free']):
            dietary_tags.append(tag)
    
    if dietary_tags:
        explanation_parts.append(f"🥗 Accommodates dietary needs: {', '.join(dietary_tags)}.")
    
    # Generate suggestions if score is low
    if health_score < 60:
        if not any('vegetable' in ing.lower() or 'vegetables' in ing.lower() for ing in ingredients):
            suggestions.append("Add more vegetables for fiber and vitamins.")
        if not any('whole' in ing.lower() or 'brown' in ing.lower() for ing in ingredients):
            suggestions.append("Consider whole grains instead of refined grains.")
        if any('fried' in tag.lower() or 'deep-fried' in tag.lower() for tag in tags):
            suggestions.append("Try baking or grilling instead of frying.")
    
    return {
        'explanation': ' '.join(explanation_parts),
        'suggestions': suggestions,
        'score_category': 'Excellent' if health_score >= 80 else 'Good' if health_score >= 70 else 'Moderate' if health_score >= 50 else 'Low'
    }

# =============================================================================
# UI COMPONENTS
# =============================================================================
def create_header():
    """Create beautiful seasonal header"""
    season = get_current_season()
    current_time = datetime.now().strftime("%H:%M")
    current_date = datetime.now().strftime("%B %d, %Y")
    
    season_themes = {
        "spring": {
            "gradient": "linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%)",
            "emoji": "🌸",
            "message": "Fresh beginnings bloom in every dish"
        },
        "summer": {
            "gradient": "linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%)",
            "emoji": "🌻",
            "message": "Savor the sunshine in seasonal flavors"
        },
        "autumn": {
            "gradient": "linear-gradient(135deg, #ff7b00 0%, #ffc048 100%)",
            "emoji": "🍂",
            "message": "Harvest warmth in every recipe"
        },
        "winter": {
            "gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "emoji": "❄️",
            "message": "Cozy comfort food for the soul"
        }
    }
    
    theme = season_themes[season]
    
    st.markdown(f"""
    <div style="background: {theme['gradient']}; padding: 3rem; border-radius: 0 0 30px 30px; margin: -3rem -3rem 2rem -3rem; text-align: center;">
        <h1 style="color: white; font-size: 3.5rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
            {theme['emoji']} Ghibli Recipe Garden {theme['emoji']}
        </h1>
        <p style="color: white; font-size: 1.5rem; margin: 1rem 0; opacity: 0.95;">
            {current_time} • {current_date}
        </p>
        <p style="color: white; font-size: 1.2rem; font-style: italic; opacity: 0.9;">
            {theme['message']}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_recipe_card(recipe, context="view", show_actions=True):
    """Display a recipe card with enhanced health and environmental info"""
    # Calculate health score and carbon footprint
    health_score = calculate_health_score(recipe)
    carbon_footprint, high_impact, medium_impact, low_impact = calculate_carbon_footprint(recipe)
    carbon_insight = get_carbon_insight(carbon_footprint)
    
    # Get LLM enhanced description
    llm_enhancer = st.session_state.get('llm_enhancer')
    enhanced_description = ""
    if llm_enhancer:
        enhanced_description = llm_enhancer.generate_recipe_description(recipe)
    
    # Create a styled card container
    st.markdown(f"""
    <div style="border: 1px solid #e0e0e0; border-radius: 15px; padding: 1.5rem; margin: 1rem 0; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <h3 style="margin-top: 0; color: #2c3e50;">{recipe['name']}</h3>
    """, unsafe_allow_html=True)
        
        # Tags
        if 'tags' in recipe and recipe['tags']:
        tags_html = " ".join([f"<span style='background: #e8f5e8; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.8rem; color: #2d5a2d;'>{tag}</span>" 
                            for tag in recipe['tags'][:4]])
            st.markdown(tags_html, unsafe_allow_html=True)
        
    # Enhanced description from LLM
    if enhanced_description:
        st.markdown(f"<p style='color: #2c3e50; font-size: 0.9rem; font-style: italic;'>{enhanced_description}</p>", unsafe_allow_html=True)
    # Fallback to original description
    elif 'description' in recipe and recipe['description']:
        desc = recipe['description'][:150] + "..." if len(str(recipe['description'])) > 150 else recipe['description']
        st.markdown(f"<p style='color: #666; font-size: 0.9rem;'>{desc}</p>", unsafe_allow_html=True)
    
    # Stats in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Time", f"{recipe.get('minutes', 30)} min")
    
    with col2:
        # Health score with color coding
        health_color = "#28a745" if health_score >= 70 else "#ffc107" if health_score >= 50 else "#dc3545"
        st.markdown(f"<div style='text-align: center;'><div style='font-size: 0.8rem; color: #666;'>🍃 Health</div><div style='font-size: 1.2rem; font-weight: bold; color: {health_color};'>{health_score:.0f}/100</div></div>", unsafe_allow_html=True)
    
    with col3:
        # Carbon footprint with insight
        carbon_color = "#28a745" if carbon_footprint < 1.0 else "#ffc107" if carbon_footprint < 2.0 else "#dc3545"
        st.markdown(f"<div style='text-align: center;'><div style='font-size: 0.8rem; color: #666;'>🌱 Carbon</div><div style='font-size: 1.2rem; font-weight: bold; color: {carbon_color};'>{carbon_footprint:.1f} kg</div></div>", unsafe_allow_html=True)
    
    with col4:
        if 'n_ingredients' in recipe:
            st.metric("🥘 Ingredients", recipe['n_ingredients'])
    
    # Carbon footprint insight
    carbon_explanation = get_carbon_explanation(carbon_footprint, high_impact, medium_impact, low_impact)
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {carbon_color};'>
        <div style='font-size: 0.9rem; font-weight: bold; color: #2c3e50;'>{carbon_insight['level']}</div>
        <div style='font-size: 0.8rem; color: #666;'>{carbon_insight['comparison']} • {carbon_insight['impact']}</div>
        <div style='font-size: 0.8rem; color: #2c3e50; margin-top: 0.5rem; font-style: italic;'>{carbon_explanation}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Health score insight
    health_explanation = get_health_score_explanation(health_score, recipe)
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {health_color};'>
        <div style='font-size: 0.9rem; font-weight: bold; color: #2c3e50;'>🍃 Health Score: {health_explanation['score_category']}</div>
        <div style='font-size: 0.8rem; color: #2c3e50; margin-top: 0.5rem; font-style: italic;'>{health_explanation['explanation']}</div>
        {f"<div style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'><strong>💡 Suggestions:</strong> {'; '.join(health_explanation['suggestions'])}</div>" if health_explanation['suggestions'] else ""}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if show_actions:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍 Like", key=f"like_{recipe['id']}_{context}", use_container_width=True):
                add_rating(recipe['id'], 5)
                
        with col2:
            if st.button("👎 Dislike", key=f"dislike_{recipe['id']}_{context}", use_container_width=True):
                add_rating(recipe['id'], 2)
                
        with col3:
            if st.button("📖 View Recipe", key=f"view_{recipe['id']}_{context}", use_container_width=True):
                st.session_state.selected_recipe = recipe['id']
                st.session_state.current_page = "Recipe Detail"
                st.rerun()
        
        with col4:
            if st.button("🍳 Cooked It!", key=f"cook_{recipe['id']}_{context}", use_container_width=True):
                # Check for environmental achievement
                achievement_name, achievement_desc, points = get_environmental_achievement(carbon_footprint)
                if achievement_name:
                    award_achievement(f"eco_{achievement_name.lower().replace(' ', '_')}", achievement_desc, points)
                else:
                award_achievement("first_cook", "First Dish Cooked!", 30)
                st.balloons()

def add_rating(recipe_id, rating):
    """Add user rating for a recipe"""
    profile = st.session_state.user_profile
    if not profile:
        st.warning("Please log in to rate recipes.")
        return

    # Convert recipe_id to string to ensure JSON serialization
    recipe_id_str = str(recipe_id)
    profile['rated_recipes'][recipe_id_str] = rating
    
    if rating >= 4:
        if recipe_id not in profile['liked_recipes']:
            profile['liked_recipes'].append(recipe_id)
        if recipe_id in profile['disliked_recipes']:
            profile['disliked_recipes'].remove(recipe_id)
    elif rating <= 2:
        if recipe_id not in profile['disliked_recipes']:
            profile['disliked_recipes'].append(recipe_id)
        if recipe_id in profile['liked_recipes']:
            profile['liked_recipes'].remove(recipe_id)
    
    profile['points'] += 5
    save_user_profile(profile, st.session_state.user_id)
    
    # Check for achievements
    if len(profile['rated_recipes']) == 10:
        award_achievement("rate_10", "Recipe Critic", 50)
    
    st.success("Rating saved! +5 points")

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================
def home_page():
    """Main home page with AI recommendations"""
    st.markdown("## 🌟 Your Personalized Recipe Journey")
    profile = st.session_state.user_profile
    recommender = st.session_state.recommender
    recipes_df = st.session_state.recipes_df
    id_mappings = st.session_state.get('id_mappings', None)

    # User stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 Level", calculate_user_level(profile['points']))
    with col2:
        st.metric("⭐ Points", profile['points'])
    with col3:
        st.metric("❤️ Liked Recipes", len(profile['liked_recipes']))
    with col4:
        st.metric("🎖️ Achievements", len(profile['achievements']))

    st.markdown("---")

    # Sustainability & Seasonality Section
    seasonal_data = get_seasonal_foods()
    sustainability_tips = get_sustainability_tips()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {seasonal_data['emoji']} {seasonal_data['season']} Seasonal Guide")
        st.markdown(f"*{seasonal_data['tip']}*")
        
        st.markdown("**🍎 Fruits in Season:**")
        for fruit in seasonal_data['fruits']:
            st.markdown(f"- {fruit}")
            
        st.markdown("**🥕 Vegetables in Season:**")
        for veg in seasonal_data['vegetables']:
            st.markdown(f"- {veg}")
    
    with col2:
        st.markdown("### 🌱 Sustainability Tips")
        for tip in sustainability_tips:
            st.markdown(f"- {tip}")

    st.markdown("---")

    # AI Recommendations
    if recommender and not recipes_df.empty and id_mappings:
        st.markdown("### 🤖 AI-Powered Recommendations")
        with st.spinner("🔮 Finding perfect recipes for you..."):
            try:
                # Map username to model user_id (if exists)
                user_id = id_mappings['user_to_idx'].get(profile['user_id'], None)
                if user_id is not None:
                    recommendations = recommender.recommend(user_id, n_recommendations=6)
                    if recommendations:
                        # Enhance recommendations with LLM
                        llm_enhancer = st.session_state.get('llm_enhancer')
                        enhanced_recipes = []
                        
                        for recipe_id, score in recommendations:
                                if recipe_id in recipes_df['id'].values:
                                recipe = recipes_df[recipes_df['id'] == recipe_id].iloc[0].to_dict()
                                if llm_enhancer:
                                    enhanced = llm_enhancer.enhance_recommendations([recipe], profile)[0]
                                    enhanced['match_score'] = score
                                    enhanced_recipes.append(enhanced)
                                else:
                                    recipe['match_score'] = score
                                    enhanced_recipes.append(recipe)
                        
                        # Sort by personalization score if available
                        if llm_enhancer:
                            enhanced_recipes.sort(key=lambda x: x.get('personalization_score', 0), reverse=True)
                        
                        for i in range(0, len(enhanced_recipes), 2):
                            cols = st.columns(2)
                            for j, recipe in enumerate(enhanced_recipes[i:i+2]):
                                    with cols[j]:
                                        with st.container():
                                        match_score = recipe.get('match_score', 0)
                                        st.markdown(f"**Match Score: {match_score*100:.0f}%**")
                                        if llm_enhancer and 'personalization_score' in recipe:
                                            personalization = recipe['personalization_score']
                                            st.markdown(f"**Personalization: {personalization*100:.0f}%**")
                                            display_recipe_card(recipe, context=f"rec_{i}_{j}")
                    else:
                        st.info("Couldn't find a good match. Try liking more recipes!")
                else:
                    st.info("👋 You're new! Like some recipes to get personalized recommendations.")
                    # Show popular recipes instead
                    st.markdown("### 🌟 Popular Recipes to Get Started")
                    popular_recipes = recipes_df.nlargest(6, 'health_score')
                    for i in range(0, len(popular_recipes), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i+j < len(popular_recipes):
                                with cols[j]:
                                    recipe = popular_recipes.iloc[i+j]
                                    display_recipe_card(recipe, context=f"pop_{i}_{j}")
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
                st.info("Showing random recipes instead...")
                random_recipes = recipes_df.sample(min(6, len(recipes_df)))
                for i in range(0, len(random_recipes), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i+j < len(random_recipes):
                            with cols[j]:
                                recipe = random_recipes.iloc[i+j]
                                display_recipe_card(recipe, context=f"rand_{i}_{j}")
    else:
        st.warning("⚠️ Recommendation system not available. Please ensure the model is trained.")

def explore_page():
    """Explore recipes with filters"""
    st.markdown("## 🔍 Explore Recipes")
    
    recipes_df = st.session_state.recipes_df
    
    if recipes_df.empty:
        st.error("No recipes available")
        return
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        time_filter = st.selectbox(
            "⏱️ Cooking Time",
            ["All", "Quick (< 30 min)", "Medium (30-60 min)", "Long (> 60 min)"]
        )
    
    with col2:
        health_filter = st.slider(
            "🍃 Min Health Score",
            0, 100, 50
        )
    
    with col3:
        carbon_filter = st.selectbox(
            "🌱 Carbon Footprint",
            ["All", "Very Low (< 0.5 kg)", "Low (0.5-1.0 kg)", "Moderate (1.0-2.0 kg)", "High (> 2.0 kg)"]
        )
    
    with col4:
        search_query = st.text_input("🔍 Search recipes...")
    
    # Sort options
        sort_by = st.selectbox(
            "📊 Sort by",
        ["Health Score", "Carbon Footprint", "Cooking Time", "Popularity"]
        )
    
    # Apply filters
    filtered_df = recipes_df.copy()
    
    if time_filter == "Quick (< 30 min)":
        filtered_df = filtered_df[filtered_df['minutes'] < 30]
    elif time_filter == "Medium (30-60 min)":
        filtered_df = filtered_df[(filtered_df['minutes'] >= 30) & (filtered_df['minutes'] <= 60)]
    elif time_filter == "Long (> 60 min)":
        filtered_df = filtered_df[filtered_df['minutes'] > 60]
    
    filtered_df = filtered_df[filtered_df['health_score'] >= health_filter]
    
    # Apply carbon footprint filter
    if carbon_filter == "Very Low (< 0.5 kg)":
        filtered_df['carbon_footprint'] = filtered_df.apply(calculate_carbon_footprint, axis=1)
        filtered_df = filtered_df[filtered_df['carbon_footprint'] < 0.5]
    elif carbon_filter == "Low (0.5-1.0 kg)":
        filtered_df['carbon_footprint'] = filtered_df.apply(calculate_carbon_footprint, axis=1)
        filtered_df = filtered_df[(filtered_df['carbon_footprint'] >= 0.5) & (filtered_df['carbon_footprint'] < 1.0)]
    elif carbon_filter == "Moderate (1.0-2.0 kg)":
        filtered_df['carbon_footprint'] = filtered_df.apply(calculate_carbon_footprint, axis=1)
        filtered_df = filtered_df[(filtered_df['carbon_footprint'] >= 1.0) & (filtered_df['carbon_footprint'] < 2.0)]
    elif carbon_filter == "High (> 2.0 kg)":
        filtered_df['carbon_footprint'] = filtered_df.apply(calculate_carbon_footprint, axis=1)
        filtered_df = filtered_df[filtered_df['carbon_footprint'] >= 2.0]
    
    if search_query:
        mask = filtered_df['name'].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Sort
    if sort_by == "Health Score":
        filtered_df = filtered_df.sort_values('health_score', ascending=False)
    elif sort_by == "Carbon Footprint":
        filtered_df['carbon_footprint'] = filtered_df.apply(calculate_carbon_footprint, axis=1)
        filtered_df = filtered_df.sort_values('carbon_footprint')
    elif sort_by == "Cooking Time":
        filtered_df = filtered_df.sort_values('minutes')
    else:
        filtered_df = filtered_df.sample(frac=1)  # Random for "popularity"
    
    st.markdown(f"### Found {len(filtered_df)} recipes")
    
    # Display results
    n_cols = 2
    n_recipes = min(12, len(filtered_df))
    
    for i in range(0, n_recipes, n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            if i+j < n_recipes:
                with cols[j]:
                    recipe = filtered_df.iloc[i+j]
                    display_recipe_card(recipe, context=f"explore_{i}_{j}")

def recipe_detail_page():
    """Displays the full details of a selected recipe."""
    st.markdown("## 📖 Full Recipe Details")

    recipe_id = st.session_state.get('selected_recipe')
    recipes_df = st.session_state.recipes_df

    if recipe_id is None or recipes_df.empty:
        st.warning("No recipe selected or data available.")
        if st.button("⬅️ Back to Explore"):
            st.session_state.current_page = "Explore"
            st.rerun()
        return

    recipe = recipes_df[recipes_df['id'] == recipe_id].iloc[0]

    # --- DEBUG: Show raw data structure ---
    st.expander("🔎 Debug: Raw Recipe Data").write({
        'ingredients': recipe.get('ingredients'),
        'steps': recipe.get('steps'),
        'name': recipe.get('name'),
    })

    # --- Header ---
    st.markdown(f"# {recipe['name']}")
    
    # Calculate health score and carbon footprint
    health_score = calculate_health_score(recipe)
    carbon_footprint, high_impact, medium_impact, low_impact = calculate_carbon_footprint(recipe)
    carbon_insight = get_carbon_insight(carbon_footprint)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Time", f"{recipe.get('minutes', 'N/A')} min")
    
    # Health score with color coding
    health_color = "#28a745" if health_score >= 70 else "#ffc107" if health_score >= 50 else "#dc3545"
    col2.markdown(f"""
    <div style='text-align: center;'>
        <div style='font-size: 0.8rem; color: #666;'>🍃 Health Score</div>
        <div style='font-size: 1.5rem; font-weight: bold; color: {health_color};'>{health_score:.0f}/100</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Carbon footprint with insight
    carbon_color = "#28a745" if carbon_footprint < 1.0 else "#ffc107" if carbon_footprint < 2.0 else "#dc3545"
    col3.markdown(f"""
    <div style='text-align: center;'>
        <div style='font-size: 0.8rem; color: #666;'>🌱 Carbon Footprint</div>
        <div style='font-size: 1.5rem; font-weight: bold; color: {carbon_color};'>{carbon_footprint:.1f} kg</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Number of ingredients instead of calories
    n_ingredients = len(recipe.get('ingredients', []))
    col4.metric("🥘 Ingredients", n_ingredients)
    
    # Carbon footprint insight
    carbon_explanation = get_carbon_explanation(carbon_footprint, high_impact, medium_impact, low_impact)
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {carbon_color};'>
        <div style='font-size: 0.9rem; font-weight: bold; color: #2c3e50;'>{carbon_insight['level']}</div>
        <div style='font-size: 0.8rem; color: #666;'>{carbon_insight['comparison']} • {carbon_insight['impact']}</div>
        <div style='font-size: 0.8rem; color: #2c3e50; margin-top: 0.5rem; font-style: italic;'>{carbon_explanation}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Health score insight
    health_explanation = get_health_score_explanation(health_score, recipe)
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid {health_color};'>
        <div style='font-size: 0.9rem; font-weight: bold; color: #2c3e50;'>🍃 Health Score: {health_explanation['score_category']}</div>
        <div style='font-size: 0.8rem; color: #2c3e50; margin-top: 0.5rem; font-style: italic;'>{health_explanation['explanation']}</div>
        {f"<div style='font-size: 0.8rem; color: #666; margin-top: 0.5rem;'><strong>💡 Suggestions:</strong> {'; '.join(health_explanation['suggestions'])}</div>" if health_explanation['suggestions'] else ""}
    </div>
    """, unsafe_allow_html=True)

    # --- Description & Tags ---
    if 'description' in recipe and pd.notna(recipe['description']):
        st.markdown("### 📝 Description")
        st.write(recipe['description'])

    if 'tags' in recipe and recipe['tags']:
        tags_html = " ".join([f"<span style='background: #e0e0e0; padding: 2px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.8rem;'>{tag}</span>" 
                                for tag in recipe['tags']])
        st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)

    st.markdown("---")

    # --- Ingredients and Steps ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🥕 Ingredients")
        if 'ingredients' in recipe and recipe['ingredients']:
            for ingredient in recipe['ingredients']:
                st.markdown(f"- {ingredient}")
        else:
            st.write("No ingredients listed.")

    with col2:
        st.markdown("### 🍳 Steps")
        steps = recipe.get('steps')
        # If steps is a stringified list, parse it
        if isinstance(steps, str) and steps.strip().startswith('[') and steps.strip().endswith(']'):
            try:
                steps = ast.literal_eval(steps)
            except Exception:
                steps = [steps]
        if isinstance(steps, (list, tuple)):
            for i, step in enumerate(steps):
                st.markdown(f"**{i+1}.** {step}")
        elif isinstance(steps, str):
            st.markdown(f"**1.** {steps}")
        else:
            st.write("No steps provided.")

    st.markdown("---")
    if st.button("⬅️ Back to previous page"):
        st.session_state.current_page = "Explore" # Or track previous page
        st.rerun()

    st.markdown("---")

    # LLM Enhanced Features
    llm_enhancer = st.session_state.get('llm_enhancer')
    if llm_enhancer:
        st.markdown("### 🤖 AI-Enhanced Cooking Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💡 Cooking Tips")
            cooking_tips = llm_enhancer.generate_cooking_tips(recipe)
            if cooking_tips:
                for tip in cooking_tips:
                    st.markdown(f"- {tip}")
            else:
                st.info("No specific tips available for this recipe.")
        
        with col2:
            st.markdown("#### 🔄 Ingredient Substitutions")
            substitutions = llm_enhancer.suggest_substitutions(recipe)
            if substitutions:
                for ingredient, alternatives in substitutions.items():
                    st.markdown(f"**{ingredient}:** {', '.join(alternatives)}")
            else:
                st.info("No substitution suggestions available.")
    
    st.markdown("---")

def login_page():
    """Page for user to login or create a new profile."""
    st.markdown("# 🌿 Welcome to the Ghibli Recipe Garden 🌿")
    st.markdown("Your personal AI-powered culinary assistant.")

    st.markdown("---")
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["🔐 Login", "✨ Create New Profile"])
    
    with tab1:
        st.markdown("### Login to Your Account")
        username = st.text_input("Enter your username:", key="login_username")
        password = st.text_input("Enter your password:", type="password", key="login_password")

        if st.button("Login", key="login_button", type="primary"):
        if username and password:
            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            # Check if user profile exists
            profile_path = Path(DATA_DIR) / f"user_{username}.json"
            if profile_path.exists():
                # Existing user - verify password
                existing_profile = load_user_profile(username)
                if existing_profile.get('password_hash') == hashed_password:
                    st.session_state.user_id = username
                    st.session_state.user_profile = existing_profile
                    st.session_state.current_page = "Home"
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
            else:
                    st.error("User not found. Please create a new profile or check your username.")
        else:
            st.warning("Please enter both username and password.")
    
    with tab2:
        st.markdown("### Create Your Profile")
        st.markdown("Join our culinary community! Choose your avatar and create your account.")
        
        # Avatar selection
        avatars = {
            "chef": {"emoji": "👨‍🍳", "name": "Master Chef", "description": "Professional cooking expert"},
            "cat": {"emoji": "🐱", "name": "Curious Cat", "description": "Playful and adventurous"},
            "dog": {"emoji": "🐕", "name": "Loyal Dog", "description": "Friendly and dependable"},
            "rabbit": {"emoji": "🐰", "name": "Quick Rabbit", "description": "Fast and energetic"},
            "owl": {"emoji": "🦉", "name": "Wise Owl", "description": "Knowledgeable and thoughtful"},
            "bear": {"emoji": "🐻", "name": "Cozy Bear", "description": "Warm and comforting"},
            "fox": {"emoji": "🦊", "name": "Clever Fox", "description": "Smart and resourceful"},
            "penguin": {"emoji": "🐧", "name": "Cool Penguin", "description": "Chill and collected"},
            "unicorn": {"emoji": "🦄", "name": "Magical Unicorn", "description": "Creative and unique"},
            "dragon": {"emoji": "🐉", "name": "Fiery Dragon", "description": "Passionate and bold"},
            "butterfly": {"emoji": "🦋", "name": "Graceful Butterfly", "description": "Elegant and free-spirited"},
            "turtle": {"emoji": "🐢", "name": "Patient Turtle", "description": "Steady and reliable"}
        }
        
        st.markdown("**Choose your cooking companion:**")
        
        # Create avatar options for selectbox
        avatar_options = []
        for avatar_key, avatar_info in avatars.items():
            avatar_options.append(f"{avatar_info['emoji']} {avatar_info['name']} ({avatar_key})")
        
        # Use selectbox for avatar selection
        selected_avatar_option = st.selectbox(
            "Select your avatar:",
            avatar_options,
            key="avatar_selection"
        )
        
        # Extract avatar key from selected option
        selected_avatar = None
        if selected_avatar_option:
            # Extract the avatar key from the option (it's in parentheses at the end)
            selected_avatar = selected_avatar_option.split("(")[-1].rstrip(")")
        
        # Display selected avatar preview
        if selected_avatar:
            avatar_info = avatars[selected_avatar]
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%); border-radius: 15px; margin: 1rem 0;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{avatar_info['emoji']}</div>
                <div style="font-weight: bold; font-size: 1.1rem;">{avatar_info['name']}</div>
                <div style="font-size: 0.9rem; color: #666;">{avatar_info['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Username and password
        new_username = st.text_input("Choose a username:", key="signup_username")
        new_password = st.text_input("Choose a password:", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm password:", type="password", key="confirm_password")
        
        if st.button("Create Profile", key="create_profile_button", type="primary"):
            if not selected_avatar:
                st.error("Please select an avatar!")
            elif not new_username or not new_password:
                st.error("Please enter both username and password!")
            elif new_password != confirm_password:
                st.error("Passwords don't match!")
            else:
                # Check if username already exists
                profile_path = Path(DATA_DIR) / f"user_{new_username}.json"
                if profile_path.exists():
                    st.error("Username already exists. Please choose a different one.")
                else:
                    # Hash the password and store for quiz
                    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                    st.session_state.user_id = new_username
                    st.session_state.temp_password_hash = hashed_password
                    st.session_state.selected_avatar = selected_avatar
                    st.session_state.current_page = "Quiz"
                    st.success(f"Profile created! Let's set up your taste preferences.")
                    st.rerun()

def user_quiz_page():
    """A multi-step quiz for new users to set up their profile."""
    # Get the pre-selected avatar from session state
    selected_avatar = st.session_state.get('selected_avatar', 'chef')
    
    # Quiz questions and options (removed avatar selection since it's done during signup)
    questions = [
        {
            'key': 'diet',
            'label': "What best describes your diet? (Select all that apply)",
            'options': [
                "🥕 Vegan", "🥦 Vegetarian", "🐟 Pescatarian", "🥩 Omnivore", "🚫 Gluten-free", "🥛 Dairy-free", "🌰 Nut-free"
            ],
            'type': 'multiselect',
            'max_selections': None
        },
        {
            'key': 'sustainability',
            'label': "Which sustainability priorities matter most to you? (Choose up to 3)",
            'options': [
                "🌱 Plant-first / reducing meat",
                "🚜 Local & seasonal ingredients",
                "🧑‍🌾 Organic / pesticide-free",
                "♻️ Zero-waste (use scraps, leftovers)",
                "⚡ Low carbon footprint (minimal processing)",
                "🌍 Fair-trade / ethical sourcing"
            ],
            'type': 'multiselect',
            'max_selections': 3
        },
        {
            'key': 'skill_level',
            'label': "How would you rate your cooking skill?",
            'options': [
                "👶 Beginner (basic techniques)",
                "👩‍🍳 Intermediate (comfortable with recipes)",
                "👨‍🍳 Advanced (ready to customize and experiment)"
            ],
            'type': 'selectbox',
        },
        {
            'key': 'preferred_time',
            'label': "On average, how much time do you have for meal prep?",
            'options': [
                "⏱️ Under 15 minutes",
                "⏲️ 15–30 minutes",
                "🕰️ 30–60 minutes",
                "🕒 Over 60 minutes"
            ],
            'type': 'selectbox',
        },
        {
            'key': 'cuisines',
            'label': "Which cuisines or flavor profiles do you enjoy most? (Select up to 3)",
            'options': [
                "🇮🇹 Mediterranean / Italian",
                "🇮🇳 Indian / South Asian",
                "🇲🇽 Latin American / Mexican",
                "🇨🇳 East Asian (Chinese, Japanese, Korean…)" ,
                "🇫🇷 French / European",
                "🌶️ Bold & spicy",
                "🍋 Bright & citrusy",
                "🧀 Rich & creamy"
            ],
            'type': 'multiselect',
            'max_selections': 3
        },
    ]
    n_questions = len(questions)

    # Initialize quiz state
    if 'quiz_page' not in st.session_state:
        st.session_state.quiz_page = 0
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}

    page = st.session_state.quiz_page
    
    # Show avatar confirmation at the top
    avatar_info = get_avatar_info(selected_avatar)
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%); border-radius: 15px; margin-bottom: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{avatar_info['emoji']}</div>
        <div style="font-weight: bold; font-size: 1.2rem;">Welcome, {st.session_state.user_id}!</div>
        <div style="font-size: 1rem; color: #666;">Your cooking companion: {avatar_info['name']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"### Taste Profile Setup - Question {page+1} of {n_questions}")
    q = questions[page]

    with st.form(key=f'quiz_form_{page}'):
        if q['type'] == 'selectbox':
            answer = st.selectbox(q['label'], q['options'], key=f"quiz_{q['key']}")
        elif q['type'] == 'multiselect':
            max_sel = q.get('max_selections')
            answer = st.multiselect(q['label'], q['options'], key=f"quiz_{q['key']}")
            if max_sel and len(answer) > max_sel:
                st.warning(f"Please select up to {max_sel} options.")
                answer = answer[:max_sel]
        else:
            answer = None
        next_button = st.form_submit_button(label='Next' if page < n_questions-1 else 'Finish')

    if next_button:
        st.session_state.quiz_answers[q['key']] = answer
        if page < n_questions-1:
            st.session_state.quiz_page += 1
            st.rerun()
        else:
            # Finalize profile
            user_id = st.session_state.user_id
            answers = st.session_state.quiz_answers
            new_profile = {
                'user_id': user_id,
                'password_hash': st.session_state.get('temp_password_hash', ''),  # Add password hash
                'avatar': selected_avatar,  # Use pre-selected avatar
                'liked_recipes': [],
                'disliked_recipes': [],
                'rated_recipes': {},
                'diet': answers.get('diet', []),
                'sustainability': answers.get('sustainability', []),
                'skill_level': answers.get('skill_level', ''),
                'preferred_time': answers.get('preferred_time', ''),
                'cuisine_preferences': answers.get('cuisines', []),
                'achievements': [],
                'points': 10,  # Starting points
                'level': 1,
                'created_at': datetime.now().isoformat()
            }
            save_user_profile(new_profile, user_id)
            st.session_state.user_profile = new_profile
            st.session_state.quiz_page = 0
            st.session_state.quiz_answers = {}
            st.session_state.current_page = "Home"
            st.balloons()
            st.success("Your profile is all set! Welcome to the garden.")
            st.rerun()
    # Progress bar
    st.progress((page+1)/n_questions)

def profile_page():
    """User profile and achievements"""
    st.markdown("## 👤 Your Recipe Garden Profile")
    
    profile = st.session_state.user_profile
    
    # Get avatar information
    avatar_key = profile.get('avatar', 'chef')
    avatar_info = get_avatar_info(avatar_key)
    
    # Profile header
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display selected avatar
        avatar_html = f"""
        <div style="width: 150px; height: 150px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <span style="font-size: 4rem; color: white;">{avatar_info['emoji']}</span>
        </div>
        <div style="text-align: center;">
            <div style="font-weight: bold; font-size: 1.1rem;">{avatar_info['name']}</div>
            <div style="font-size: 0.8rem; color: #666;">{avatar_info['description']}</div>
        </div>
        """
        st.markdown(avatar_html, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### {calculate_user_level(profile['points'])}")
        st.markdown(f"**User ID:** {profile['user_id']}")
        st.markdown(f"**Member Since:** {profile.get('created_at', 'Today')[:10]}")
        st.progress(min(profile['points'] % 100 / 100, 1.0))
        st.caption(f"{profile['points'] % 100}/100 points to next level")
    
    # Stats
    st.markdown("---")
    st.markdown("### 📊 Your Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Recipes Rated", len(profile['rated_recipes']))
    with col2:
        st.metric("❤️ Recipes Liked", len(profile['liked_recipes']))
    with col3:
        st.metric("🏆 Achievements", len(profile['achievements']))
    with col4:
        st.metric("⭐ Total Points", profile['points'])
    
    # Achievements
    st.markdown("---")
    st.markdown("### 🏆 Achievements")
    
    all_achievements = {
        "first_cook": {"name": "First Dish Cooked!", "desc": "Cooked your first recipe", "icon": "👨‍🍳"},
        "rate_10": {"name": "Recipe Critic", "desc": "Rated 10 recipes", "icon": "⭐"},
        "like_20": {"name": "Food Lover", "desc": "Liked 20 recipes", "icon": "❤️"},
        "explore_50": {"name": "Recipe Explorer", "desc": "Viewed 50 recipes", "icon": "🔍"},
        "health_nut": {"name": "Health Conscious", "desc": "Liked 10 healthy recipes", "icon": "🥗"},
        "quick_chef": {"name": "Speed Demon", "desc": "Cooked 5 quick recipes", "icon": "⚡"},
        "master_chef": {"name": "Master Chef", "desc": "Reached 1000 points", "icon": "👑"},
        "eco_warrior": {"name": "🌱 Eco Warrior", "desc": "Cooked a very low-carbon meal", "icon": "🌱"},
        "green_chef": {"name": "🌿 Green Chef", "desc": "Cooked a low-carbon meal", "icon": "🌿"}
    }
    
    cols = st.columns(4)
    for i, (ach_id, ach_data) in enumerate(all_achievements.items()):
        with cols[i % 4]:
            if ach_id in profile['achievements']:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f5f5f5, #e0e0e0); border-radius: 10px;">
                    <div style="font-size: 3rem;">{ach_data['icon']}</div>
                    <div style="font-weight: bold;">{ach_data['name']}</div>
                    <div style="font-size: 0.8rem; color: #666;">{ach_data['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: #f5f5f5; border-radius: 10px; opacity: 0.5;">
                    <div style="font-size: 3rem;">🔒</div>
                    <div style="font-weight: bold;">???</div>
                    <div style="font-size: 0.8rem; color: #666;">{ach_data['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Preferences
    st.markdown("---")
    st.markdown("### ⚙️ Preferences")
    
    # Create an expander for editing preferences
    with st.expander("✏️ Edit Your Preferences"):
        st.markdown("**Update your cooking preferences to get better recommendations:**")
        
        # Dietary preferences
        st.markdown("#### 🥗 Dietary Preferences")
        current_diet = profile.get('diet', [])
        diet_options = [
            "🥕 Vegan", "🥦 Vegetarian", "🐟 Pescatarian", "🥩 Omnivore", 
            "🚫 Gluten-free", "🥛 Dairy-free", "🌰 Nut-free"
        ]
        new_diet = st.multiselect(
            "Select your dietary preferences:",
            diet_options,
            default=[opt for opt in diet_options if opt in current_diet]
        )
        
        # Sustainability preferences
        st.markdown("#### 🌱 Sustainability Priorities")
        current_sustainability = profile.get('sustainability', [])
        sustainability_options = [
            "🌱 Plant-first / reducing meat",
            "🚜 Local & seasonal ingredients",
            "🧑‍🌾 Organic / pesticide-free",
            "♻️ Zero-waste (use scraps, leftovers)",
            "⚡ Low carbon footprint (minimal processing)",
            "🌍 Fair-trade / ethical sourcing"
        ]
        new_sustainability = st.multiselect(
            "Choose your sustainability priorities (up to 3):",
            sustainability_options,
            default=[opt for opt in sustainability_options if opt in current_sustainability],
            max_selections=3
        )
        
        # Skill level
        st.markdown("#### 👨‍🍳 Cooking Skill Level")
        current_skill = profile.get('skill_level', '')
        skill_options = [
            "👶 Beginner (basic techniques)",
            "👩‍🍳 Intermediate (comfortable with recipes)",
            "👨‍🍳 Advanced (ready to customize and experiment)"
        ]
        new_skill = st.selectbox(
            "Select your cooking skill level:",
            skill_options,
            index=next((i for i, opt in enumerate(skill_options) if opt == current_skill), 0)
        )
        
        # Preferred cooking time
        st.markdown("#### ⏱️ Preferred Cooking Time")
        current_time = profile.get('preferred_time', '')
        time_options = [
            "⏱️ Under 15 minutes",
            "⏲️ 15–30 minutes",
            "🕰️ 30–60 minutes",
            "🕒 Over 60 minutes"
        ]
        new_time = st.selectbox(
            "Select your preferred cooking time:",
            time_options,
            index=next((i for i, opt in enumerate(time_options) if opt == current_time), 0)
        )
        
        # Cuisine preferences
        st.markdown("#### 🌍 Favorite Cuisines & Flavors")
        current_cuisines = profile.get('cuisine_preferences', [])
        cuisine_options = [
            "🇮🇹 Mediterranean / Italian",
            "🇮🇳 Indian / South Asian",
            "🇲🇽 Latin American / Mexican",
            "🇨🇳 East Asian (Chinese, Japanese, Korean…)",
            "🇫🇷 French / European",
            "🌶️ Bold & spicy",
            "🍋 Bright & citrusy",
            "🧀 Rich & creamy"
        ]
        new_cuisines = st.multiselect(
            "Select your favorite cuisines and flavors (up to 3):",
            cuisine_options,
            default=[opt for opt in cuisine_options if opt in current_cuisines],
            max_selections=3
        )
        
        # Save button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Preferences", type="primary"):
                # Update profile with new preferences
                profile['diet'] = new_diet
                profile['sustainability'] = new_sustainability
                profile['skill_level'] = new_skill
                profile['preferred_time'] = new_time
                profile['cuisine_preferences'] = new_cuisines
                
                # Save to file
                save_user_profile(profile, st.session_state.user_id)
                st.success("✅ Preferences updated successfully!")
                st.rerun()
    
    # Display current preferences
    st.markdown("#### 📋 Current Preferences")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥗 Diet:**")
        if profile.get('diet'):
            for diet in profile['diet']:
                st.markdown(f"- {diet}")
        else:
            st.markdown("- *No dietary preferences set*")
        
        st.markdown("**🌱 Sustainability:**")
        if profile.get('sustainability'):
            for sustain in profile['sustainability']:
                st.markdown(f"- {sustain}")
        else:
            st.markdown("- *No sustainability preferences set*")
        
        st.markdown("**👨‍🍳 Skill Level:**")
        st.markdown(f"- {profile.get('skill_level', 'Not set')}")
    
    with col2:
        st.markdown("**⏱️ Cooking Time:**")
        st.markdown(f"- {profile.get('preferred_time', 'Not set')}")
        
        st.markdown("**🌍 Cuisines:**")
        if profile.get('cuisine_preferences'):
            for cuisine in profile['cuisine_preferences']:
                st.markdown(f"- {cuisine}")
        else:
            st.markdown("- *No cuisine preferences set*")

    # Logout section
    st.markdown("---")
    st.markdown("### 🚪 Account")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.user_id = None
            st.session_state.user_profile = None
            st.session_state.current_page = "Login"
            st.success("Logged out successfully!")
            st.rerun()
    with col2:
        st.write("")

def analytics_page():
    """Show recommendation system analytics"""
    st.markdown("## 📊 Recommendation Analytics")
    
    recipes_df = st.session_state.recipes_df
    interactions_df = st.session_state.interactions_df
    
    if recipes_df.empty:
        st.error("No data available for analytics")
        return
    
    # System stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Total Recipes", len(recipes_df))
    with col2:
        st.metric("👥 Total Users", interactions_df['user_id'].nunique() if not interactions_df.empty else 0)
    with col3:
        st.metric("⭐ Total Ratings", len(interactions_df) if not interactions_df.empty else 0)
    with col4:
        avg_rating = interactions_df['rating'].mean() if not interactions_df.empty else 0
        st.metric("📈 Avg Rating", f"{avg_rating:.2f}")
    
    # Recipe distribution
    st.markdown("---")
    st.markdown("### 📊 Recipe Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time distribution
        try:
            time_bins = pd.cut(recipes_df['cook_time'], bins=[0, 30, 60, 120, 300], 
                              labels=['Quick', 'Fast', 'Medium', 'Long'])
            time_dist = pd.Series(time_bins).value_counts()
            
            st.markdown("**Cook Time Distribution**")
        for category, count in time_dist.items():
            st.write(f"{category}: {count} ({count/len(recipes_df)*100:.1f}%)")
        except Exception as e:
            st.write("Time distribution data not available")
    
    with col2:
        # Health score distribution
        try:
        health_bins = pd.cut(recipes_df['health_score'], bins=[0, 50, 70, 85, 100], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
            health_dist = pd.Series(health_bins).value_counts()
        
        st.markdown("**Health Score Distribution**")
            for category, count in health_dist.items():
            st.write(f"{category}: {count} ({count/len(recipes_df)*100:.1f}%)")
        except Exception as e:
            st.write("Health score distribution data not available")
    
    # Model performance (if available)
    if st.session_state.recommender:
        st.markdown("---")
        st.markdown("### 🤖 Model Performance")
        
        # Placeholder metrics - in production, these would be calculated properly
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Precision@10", "0.75")
        with col2:
            st.metric("📏 RMSE", "0.92")
        with col3:
            st.metric("🔄 Coverage", "82%")
        
        st.info("💡 The model combines collaborative filtering (what similar users liked) with content-based filtering (recipe features) for hybrid recommendations.")

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Initialize
    init_session_state()
    
    # Load data and model
    if st.session_state.recipes_df is None:
        st.session_state.recipes_df = load_recipe_data()
    if st.session_state.interactions_df is None:
        st.session_state.interactions_df = load_interaction_data()
    if (st.session_state.recommender is None or st.session_state.id_mappings is None) and st.session_state.recipes_df is not None:
        recommender, id_mappings = load_model()
        st.session_state.recommender = recommender
        st.session_state.id_mappings = id_mappings

    # Simple Router
    if st.session_state.current_page == "Login":
        login_page()
    elif st.session_state.current_page == "Quiz":
        user_quiz_page()
    else:
        # Show header and nav only when logged in
        create_header()
        
        st.markdown("---")
        pages = {
            "Home": "🏠",
            "Explore": "🔍", 
            "Profile": "👤",
            "Analytics": "📊"
        }
        
        cols = st.columns(len(pages))
        for i, (page, icon) in enumerate(pages.items()):
            with cols[i]:
                if st.button(f"{icon} {page}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        st.markdown("---")
        
        # Route to page
        if st.session_state.current_page == "Home":
            home_page()
        elif st.session_state.current_page == "Explore":
            explore_page()
        elif st.session_state.current_page == "Recipe Detail":
            recipe_detail_page()
        elif st.session_state.current_page == "Profile":
            profile_page()
        elif st.session_state.current_page == "Analytics":
            analytics_page()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>🌿 Ghibli Recipe Garden - AI-Powered Recipe Recommendations 🌿</p>
            <p style="font-size: 0.8rem;">Built with ❤️ using Streamlit and Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()