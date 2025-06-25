"""
Enhanced Recipe Recommendation App with Preference Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, date
import json
import os
from recommendation_models import HybridRecommender
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
# PREFERENCE-BASED RECOMMENDATION SYSTEM
# =============================================================================
def get_preference_based_recommendations(recipes_df, profile, n_recommendations=12):
    """Get recommendations based on user preferences from questionnaire"""
    if recipes_df.empty:
        return []
    
    # Calculate preference scores for each recipe
    preference_scores = []
    
    for _, recipe in recipes_df.iterrows():
        score = 0
        recipe_dict = recipe.to_dict()
        
        # Cuisine preferences
        cuisine_prefs = profile.get('cuisine_preferences', [])
        recipe_text = f"{recipe['name']} {' '.join(recipe.get('ingredients', []))} {' '.join(recipe.get('tags', []))}"
        recipe_text_lower = recipe_text.lower()
        
        for cuisine_pref in cuisine_prefs:
            if 'spicy' in cuisine_pref.lower() and any(spice in recipe_text_lower for spice in ['spicy', 'hot', 'chili', 'pepper', 'cayenne', 'jalapeno', 'paprika', 'curry powder', 'red pepper flakes', 'sriracha', 'tabasco']):
                score += 30
            elif 'italian' in cuisine_pref.lower() and any(italian in recipe_text_lower for italian in ['pasta', 'pizza', 'risotto', 'parmesan', 'basil', 'oregano', 'mozzarella', 'tomato sauce', 'prosciutto', 'balsamic']):
                score += 25
            elif 'mexican' in cuisine_pref.lower() and any(mexican in recipe_text_lower for mexican in ['taco', 'burrito', 'enchilada', 'salsa', 'cilantro', 'lime', 'tortilla', 'avocado', 'queso', 'pico de gallo']):
                score += 25
            elif 'indian' in cuisine_pref.lower() and any(indian in recipe_text_lower for indian in ['curry', 'naan', 'tandoori', 'masala', 'dal', 'turmeric', 'cumin', 'cardamom', 'garam masala', 'chutney']):
                score += 25
            elif 'chinese' in cuisine_pref.lower() and any(chinese in recipe_text_lower for chinese in ['stir-fry', 'dim sum', 'soy sauce', 'ginger', 'sesame', 'bok choy', 'rice', 'wonton', 'dumpling']):
                score += 25
            elif 'japanese' in cuisine_pref.lower() and any(japanese in recipe_text_lower for japanese in ['sushi', 'ramen', 'tempura', 'miso', 'wasabi', 'nori', 'dashi', 'teriyaki', 'udon']):
                score += 25
            elif 'french' in cuisine_pref.lower() and any(french in recipe_text_lower for french in ['ratatouille', 'coq au vin', 'quiche', 'bechamel', 'herbs de provence', 'shallots', 'dijon', 'tarragon']):
                score += 25
            elif 'mediterranean' in cuisine_pref.lower() and any(med in recipe_text_lower for med in ['hummus', 'falafel', 'olive oil', 'feta', 'oregano', 'chickpeas', 'tahini', 'za\'atar', 'sumac']):
                score += 25
            elif 'citrusy' in cuisine_pref.lower() and any(citrus in recipe_text_lower for citrus in ['lemon', 'lime', 'orange', 'citrus', 'zest', 'juice', 'yuzu', 'bergamot']):
                score += 20
            elif 'creamy' in cuisine_pref.lower() and any(creamy in recipe_text_lower for creamy in ['cream', 'cheese', 'butter', 'milk', 'sour cream', 'yogurt', 'mayonnaise', 'alfredo', 'béchamel']):
                score += 20
        
        # Dietary preferences
        diet_prefs = profile.get('diet', [])
        for diet_pref in diet_prefs:
            if 'vegan' in diet_pref.lower() and 'vegan' in recipe_text_lower:
                score += 20
            elif 'vegetarian' in diet_pref.lower() and 'vegetarian' in recipe_text_lower:
                score += 15
            elif 'gluten-free' in diet_pref.lower() and 'gluten-free' in recipe_text_lower:
                score += 15
            elif 'dairy-free' in diet_pref.lower() and not any(dairy in recipe_text_lower for dairy in ['milk', 'cheese', 'butter', 'cream', 'yogurt']):
                score += 15
        
        # Cooking time preferences
        time_pref = profile.get('preferred_time', '')
        minutes = recipe.get('minutes', 30)
        
        if 'under 15' in time_pref.lower() and minutes <= 15:
            score += 20
        elif '15–30' in time_pref.lower() and 15 <= minutes <= 30:
            score += 20
        elif '30–60' in time_pref.lower() and 30 <= minutes <= 60:
            score += 20
        elif 'over 60' in time_pref.lower() and minutes > 60:
            score += 20
        
        # Skill level preferences
        skill_level = profile.get('skill_level', '')
        if 'beginner' in skill_level.lower() and minutes <= 30:
            score += 15
        elif 'advanced' in skill_level.lower() and minutes > 60:
            score += 15
        
        # Avoid disliked recipes
        if recipe['id'] in profile.get('disliked_recipes', []):
            score -= 50
        
        # Boost liked recipes
        if recipe['id'] in profile.get('liked_recipes', []):
            score += 10
        
        preference_scores.append((recipe_dict, score))
    
    # Sort by preference score and return top recommendations
    preference_scores.sort(key=lambda x: x[1], reverse=True)
    return [(recipe, score) for recipe, score in preference_scores[:n_recommendations]]

def combine_recommendations(ml_recommendations, preference_recommendations, profile):
    """Combine ML and preference-based recommendations"""
    combined = []
    
    # Add ML recommendations with high weight
    for recipe_id, ml_score in ml_recommendations:
        combined.append({
            'recipe_id': recipe_id,
            'ml_score': ml_score,
            'preference_score': 0,
            'combined_score': ml_score * 0.7,  # ML gets 70% weight
            'source': 'ML'
        })
    
    # Add preference recommendations
    for recipe, pref_score in preference_recommendations:
        recipe_id = recipe['id']
        # Check if already in combined list
        existing = next((item for item in combined if item['recipe_id'] == recipe_id), None)
        if existing:
            existing['preference_score'] = pref_score / 100  # Normalize to 0-1
            existing['combined_score'] = existing['ml_score'] * 0.7 + (pref_score / 100) * 0.3
            existing['source'] = 'Hybrid'
    else:
            combined.append({
                'recipe_id': recipe_id,
                'ml_score': 0,
                'preference_score': pref_score / 100,
                'combined_score': (pref_score / 100) * 0.3,  # Preference gets 30% weight
                'source': 'Preferences'
            })
    
    # Sort by combined score and return unique recipes
    combined.sort(key=lambda x: x['combined_score'], reverse=True)
    seen_ids = set()
    final_recommendations = []
    
    for item in combined:
        if item['recipe_id'] not in seen_ids:
            final_recommendations.append(item)
            seen_ids.add(item['recipe_id'])
    
    return final_recommendations[:12]  # Return top 12

# =============================================================================
# SOCIAL FEATURES
# =============================================================================
def get_social_recommendations(recipes_df, profile, other_users_data):
    """Get recommendations based on what similar users liked"""
    if not other_users_data:
        return []
    
    # Find users with similar preferences
    similar_users = []
    user_cuisines = set(profile.get('cuisine_preferences', []))
    user_diet = set(profile.get('diet', []))
    
    for other_user in other_users_data:
        other_cuisines = set(other_user.get('cuisine_preferences', []))
        other_diet = set(other_user.get('diet', []))
        
        # Calculate similarity score
        cuisine_overlap = len(user_cuisines.intersection(other_cuisines))
        diet_overlap = len(user_diet.intersection(other_diet))
        
        similarity = cuisine_overlap + diet_overlap
        if similarity > 0:
            similar_users.append((other_user, similarity))
    
    # Sort by similarity and get top similar users
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Get recipes liked by similar users
    recommended_recipes = []
    for user, similarity in similar_users[:5]:  # Top 5 similar users
        for recipe_id in user.get('liked_recipes', []):
            if recipe_id in recipes_df['id'].values:
                recipe = recipes_df[recipes_df['id'] == recipe_id].iloc[0].to_dict()
                recommended_recipes.append((recipe, similarity * 10))  # Weight by similarity
    
    # Remove duplicates and sort
    seen_ids = set()
    unique_recipes = []
    for recipe, score in recommended_recipes:
        if recipe['id'] not in seen_ids:
            unique_recipes.append((recipe, score))
            seen_ids.add(recipe['id'])
    
    unique_recipes.sort(key=lambda x: x[1], reverse=True)
    return unique_recipes[:6]

def create_recipe_collections():
    """Create themed recipe collections for community sharing"""
    collections = {
        "Spicy Adventures": {
            "description": "For those who love heat!",
            "emoji": "🌶️",
            "tags": ["spicy", "hot", "chili", "pepper"]
        },
        "Quick & Healthy": {
            "description": "Nutritious meals in 30 minutes or less",
            "emoji": "⚡",
            "tags": ["quick", "healthy", "nutritious"]
        },
        "Plant-Based Delights": {
            "description": "Delicious vegan and vegetarian options",
            "emoji": "🌱",
            "tags": ["vegan", "vegetarian", "plant-based"]
        },
        "Comfort Classics": {
            "description": "Cozy, comforting dishes",
            "emoji": "🫂",
            "tags": ["comfort", "cozy", "classic"]
        }
    }
    return collections

# =============================================================================
# ENHANCED UI COMPONENTS
# =============================================================================
def display_preference_badge(recipe, profile):
    """Display badges showing why a recipe matches user preferences"""
    badges = []
    
    cuisine_prefs = profile.get('cuisine_preferences', [])
    recipe_text = f"{recipe['name']} {' '.join(recipe.get('ingredients', []))} {' '.join(recipe.get('tags', []))}"
    recipe_text_lower = recipe_text.lower()
    
    for cuisine_pref in cuisine_prefs:
        if 'spicy' in cuisine_pref.lower() and any(spice in recipe_text_lower for spice in ['spicy', 'hot', 'chili', 'pepper']):
            badges.append("🌶️ Spicy Match")
        elif 'italian' in cuisine_pref.lower() and any(italian in recipe_text_lower for italian in ['pasta', 'pizza', 'risotto']):
            badges.append("🇮🇹 Italian Match")
        elif 'mexican' in cuisine_pref.lower() and any(mexican in recipe_text_lower for mexican in ['taco', 'burrito', 'salsa']):
            badges.append("🇲🇽 Mexican Match")
    
    # Time preference match
    time_pref = profile.get('preferred_time', '')
    minutes = recipe.get('minutes', 30)
    if 'under 15' in time_pref.lower() and minutes <= 15:
        badges.append("⚡ Quick Match")
    elif '30–60' in time_pref.lower() and 30 <= minutes <= 60:
        badges.append("⏱️ Time Match")
    
    return badges

def display_enhanced_recipe_card(recipe, profile, context="view", show_actions=True):
    """Enhanced recipe card with preference matching indicators"""
    # Get preference badges
    preference_badges = display_preference_badge(recipe, profile)
    
    # Display badges
    if preference_badges:
        badge_html = " ".join([f"<span style='background: #e3f2fd; padding: 3px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.8rem; color: #1976d2;'>{badge}</span>" 
                              for badge in preference_badges[:3]])
        st.markdown(badge_html, unsafe_allow_html=True)
    
    # Call the original display function
    display_recipe_card(recipe, context, show_actions)

# =============================================================================
# MAIN ENHANCED HOME PAGE
# =============================================================================
def enhanced_home_page():
    """Enhanced home page with preference integration and social features"""
    st.markdown("## 🌟 Your Personalized Recipe Journey")
    profile = st.session_state.user_profile
    recommender = st.session_state.recommender
    recipes_df = st.session_state.recipes_df
    id_mappings = st.session_state.get('id_mappings', None)

    # User stats with preference summary
    col1, col2, col3, col4 = st.columns(4)
            with col1:
        st.metric("🏆 Level", calculate_user_level(profile['points']))
            with col2:
        st.metric("⭐ Points", profile['points'])
            with col3:
        st.metric("❤️ Liked Recipes", len(profile['liked_recipes']))
    with col4:
        st.metric("🎖️ Achievements", len(profile['achievements']))

    # Show user preferences summary
    if profile.get('cuisine_preferences') or profile.get('diet'):
        st.markdown("### 🎯 Your Taste Profile")
        col1, col2 = st.columns(2)
        with col1:
            if profile.get('cuisine_preferences'):
                st.markdown("**🌍 Cuisines:** " + ", ".join(profile['cuisine_preferences']))
            if profile.get('diet'):
                st.markdown("**🥗 Diet:** " + ", ".join(profile['diet']))
        with col2:
            if profile.get('skill_level'):
                st.markdown(f"**👨‍🍳 Skill:** {profile['skill_level']}")
            if profile.get('preferred_time'):
                st.markdown(f"**⏱️ Time:** {profile['preferred_time']}")

    st.markdown("---")

    # Enhanced Recommendations Section
    if recommender and not recipes_df.empty and id_mappings:
        st.markdown("### 🤖 Smart Recommendations")
        with st.spinner("🔮 Finding perfect recipes for you..."):
            try:
                # Get preference-based recommendations
                preference_recommendations = get_preference_based_recommendations(recipes_df, profile, 12)
                
                # Map username to model user_id (if exists)
                user_id = id_mappings['user_to_idx'].get(profile['user_id'], None)
                if user_id is not None:
                    ml_recommendations = recommender.recommend(user_id, n_recommendations=6)
                    
                    # Combine recommendations
                    if ml_recommendations and preference_recommendations:
                        combined = combine_recommendations(ml_recommendations, preference_recommendations, profile)
                        
                        # Display with source indicators
                        st.markdown("#### 🎯 Hybrid Recommendations")
                        st.info("💡 These combine ML insights with your questionnaire preferences!")
                        
                        for i in range(0, len(combined), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i+j < len(combined):
                                    with cols[j]:
                                        item = combined[i+j]
                                        recipe_id = item['recipe_id']
                                        if recipe_id in recipes_df['id'].values:
                                            recipe = recipes_df[recipes_df['id'] == recipe_id].iloc[0].to_dict()
                                            recipe['match_score'] = item['combined_score']
                                            
                                            # Show source and preference badges
                                            source_color = "#28a745" if item['source'] == 'Hybrid' else "#007bff" if item['source'] == 'ML' else "#ffc107"
                                            st.markdown(f"<div style='background: {source_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; display: inline-block; margin-bottom: 0.5rem;'>{item['source']}</div>", unsafe_allow_html=True)
                                            st.markdown(f"**Match Score: {item['combined_score']*100:.0f}%**")
                                            display_enhanced_recipe_card(recipe, profile, context=f"hybrid_{i}_{j}")
                    
                    elif preference_recommendations:
                        # Show preference-based recommendations
                        st.markdown("#### 🌟 Based on Your Preferences")
                        st.info("💡 These recommendations are based on your questionnaire answers!")
                        
                        for i in range(0, len(preference_recommendations), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i+j < len(preference_recommendations):
                                    with cols[j]:
                                        recipe, score = preference_recommendations[i+j]
                                        recipe['match_score'] = score / 100
                                        
                                        st.markdown(f"<div style='background: #ffc107; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; display: inline-block; margin-bottom: 0.5rem;'>Preferences</div>", unsafe_allow_html=True)
                                        st.markdown(f"**Preference Score: {score:.0f}**")
                                        display_enhanced_recipe_card(recipe, profile, context=f"pref_{i}_{j}")
                
            else:
                    # New user - show preference-based recommendations
                    if preference_recommendations:
                        st.markdown("#### 🌟 Welcome! Based on Your Preferences")
                        st.info("👋 You're new! These recommendations are based on your questionnaire answers.")
                        
                        for i in range(0, len(preference_recommendations), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i+j < len(preference_recommendations):
                                    with cols[j]:
                                        recipe, score = preference_recommendations[i+j]
                                        recipe['match_score'] = score / 100
                                        
                                        st.markdown(f"<div style='background: #ffc107; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; display: inline-block; margin-bottom: 0.5rem;'>Preferences</div>", unsafe_allow_html=True)
                                        st.markdown(f"**Preference Score: {score:.0f}**")
                                        display_enhanced_recipe_card(recipe, profile, context=f"new_{i}_{j}")
                    
            except Exception as e:
                st.error(f"Error getting recommendations: {e}")
                st.info("Showing preference-based recommendations instead...")
                if preference_recommendations:
                    for i in range(0, len(preference_recommendations), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i+j < len(preference_recommendations):
                                with cols[j]:
                                    recipe, score = preference_recommendations[i+j]
                                    display_enhanced_recipe_card(recipe, profile, context=f"fallback_{i}_{j}")

    # Community Collections Section
    st.markdown("---")
    st.markdown("### 👥 Community Collections")
    collections = create_recipe_collections()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌶️ Spicy Adventures")
        st.markdown("*For those who love heat!*")
        # Filter recipes for spicy collection
        spicy_recipes = recipes_df[recipes_df['name'].str.contains('spicy|hot|chili|pepper', case=False, na=False)].head(3)
        for _, recipe in spicy_recipes.iterrows():
            st.markdown(f"- **{recipe['name']}** ({recipe.get('minutes', 30)} min)")
    
    with col2:
        st.markdown("#### ⚡ Quick & Healthy")
        st.markdown("*Nutritious meals in 30 minutes or less*")
        # Filter recipes for quick healthy collection
        quick_healthy = recipes_df[(recipes_df['minutes'] <= 30) & (recipes_df['health_score'] >= 70)].head(3)
        for _, recipe in quick_healthy.iterrows():
            st.markdown(f"- **{recipe['name']}** (Health: {recipe['health_score']:.0f})")

# =============================================================================
# DESIGN IMPROVEMENTS SUGGESTIONS
# =============================================================================
def get_design_improvements():
    """Return suggestions for design improvements"""
    return {
        "UI/UX Improvements": [
            "🎨 Add dark mode toggle for better accessibility",
            "📱 Implement responsive design for mobile users",
            "🎯 Add recipe difficulty indicators (Beginner/Intermediate/Advanced)",
            "⏰ Show estimated prep time vs cook time separately",
            "📊 Add nutrition facts and ingredient substitution options",
            "🎬 Include cooking video tutorials for complex recipes",
            "📝 Add recipe notes and personal modifications feature",
            "🔄 Implement recipe scaling (servings adjustment)",
            "📅 Add meal planning calendar integration",
            "🛒 Generate shopping lists from selected recipes"
        ],
        "Social Features": [
            "👥 User profiles with cooking achievements and badges",
            "💬 Recipe reviews and comments system",
            "📸 Photo sharing of cooked dishes",
            "👨‍👩‍👧‍👦 Family recipe sharing and collaboration",
            "🏆 Cooking challenges and competitions",
            "📱 Push notifications for new recipes matching preferences",
            "🤝 Recipe recommendations from friends",
            "📊 Community cooking statistics and trends",
            "🎉 Virtual cooking parties and live sessions",
            "📚 User-generated recipe collections and cookbooks"
        ],
        "AI Enhancements": [
            "🤖 Chatbot for cooking assistance and troubleshooting",
            "📷 Image recognition for ingredient identification",
            "🎯 Personalized cooking tips based on skill level",
            "🌡️ Smart temperature and timing recommendations",
            "🥘 Ingredient substitution suggestions based on availability",
            "📊 Nutritional analysis and dietary compliance checking",
            "🎨 Recipe customization based on available ingredients",
            "📈 Learning progress tracking and skill development",
            "🔍 Advanced search with natural language queries",
            "🎵 Cooking playlist recommendations"
        ],
        "Engagement Features": [
            "🏆 Gamification with cooking levels and achievements",
            "📈 Progress tracking and cooking statistics",
            "🎯 Weekly cooking challenges and themes",
            "💝 Recipe gifting and sharing features",
            "📱 Mobile app with offline recipe access",
            "🎨 Recipe customization and personal cookbook creation",
            "📊 Cooking analytics and improvement suggestions",
            "🎉 Seasonal events and holiday recipe collections",
            "🤝 Community forums and cooking discussions",
            "📚 Educational content and cooking tutorials"
        ]
    }

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main function to run the enhanced app"""
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Login"
    
    # Simple router for demonstration
    if st.session_state.current_page == "Login":
        st.markdown("# 🌿 Enhanced Recipe Garden")
        st.markdown("### Login to see preference-based recommendations")
        username = st.text_input("Username")
        if st.button("Login"):
            st.session_state.user_id = username
            st.session_state.user_profile = {
                'user_id': username,
                'cuisine_preferences': ['🌶️ Bold & spicy', '🇮🇹 Mediterranean / Italian'],
                'diet': ['🥦 Vegetarian'],
                'skill_level': '👩‍🍳 Intermediate (comfortable with recipes)',
                'preferred_time': '⏲️ 15–30 minutes',
                'liked_recipes': [],
                'disliked_recipes': [],
                'points': 50
            }
            st.session_state.current_page = "Home"
            st.rerun()
    
    elif st.session_state.current_page == "Home":
        enhanced_home_page()
        
        # Show design improvements
        if st.button("💡 View Design Improvement Suggestions"):
            improvements = get_design_improvements()
            for category, suggestions in improvements.items():
                st.markdown(f"### {category}")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
        
        if st.button("🚪 Logout"):
            st.session_state.current_page = "Login"
            st.session_state.user_id = None
            st.session_state.user_profile = None
            st.rerun()

if __name__ == "__main__":
    main()