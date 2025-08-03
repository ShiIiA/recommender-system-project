"""
Local LLM Enhancement features
"""
import random

class LocalLLMEnhancer:
    """Local LLM enhancement for recipes"""
    
    def __init__(self):
        self.available = True
        
    def generate_recipe_description(self, recipe):
        """Generate enhanced description"""
        cuisine = recipe.get('cuisine', 'International')
        name = recipe.get('name', 'Recipe')
        
        descriptions = [
            f"A delightful {cuisine.lower()} dish that brings authentic flavors to your table.",
            f"This {cuisine.lower()} recipe combines traditional techniques with modern convenience.",
            f"Experience the rich tastes of {cuisine.lower()} cuisine with this carefully crafted recipe.",
            f"A perfect blend of ingredients that showcases the best of {cuisine.lower()} cooking."
        ]
        
        return random.choice(descriptions)
    
    def enhance_recommendations(self, recipes, profile):
        """Enhance recipe recommendations with personalization scores"""
        enhanced_recipes = []
        
        for recipe in recipes:
            enhanced_recipe = recipe.copy()
            
            # Calculate personalization score based on profile
            personalization_score = 0.5  # Base score
            
            # Check dietary preferences
            user_diet = profile.get('diet', [])
            recipe_tags = recipe.get('tags', [])
            
            for diet in user_diet:
                diet_lower = diet.lower()
                if any(diet_lower in tag.lower() for tag in recipe_tags):
                    personalization_score += 0.1
            
            # Check cuisine preferences
            user_cuisines = profile.get('cuisine_preferences', [])
            recipe_cuisine = recipe.get('cuisine', '').lower()
            
            for cuisine_pref in user_cuisines:
                if recipe_cuisine in cuisine_pref.lower():
                    personalization_score += 0.15
            
            # Check cooking time preferences
            user_time = profile.get('preferred_time', '')
            recipe_minutes = recipe.get('minutes', 30)
            
            if '15' in user_time and recipe_minutes <= 15:
                personalization_score += 0.1
            elif '30' in user_time and 15 < recipe_minutes <= 30:
                personalization_score += 0.1
            elif '60' in user_time and 30 < recipe_minutes <= 60:
                personalization_score += 0.1
            
            # Cap the score at 1.0
            enhanced_recipe['personalization_score'] = min(1.0, personalization_score)
            enhanced_recipes.append(enhanced_recipe)
        
        return enhanced_recipes
    
    def generate_cooking_tips(self, recipe):
        """Generate cooking tips for a recipe"""
        cuisine = recipe.get('cuisine', 'International').lower()
        
        general_tips = [
            "Always taste and adjust seasoning as you cook",
            "Prep all ingredients before you start cooking",
            "Use fresh ingredients for the best flavor",
            "Don't overcrowd the pan when cooking"
        ]
        
        cuisine_tips = {
            'italian': ["Save pasta water to help bind your sauce", "Don't add oil to pasta water"],
            'mexican': ["Toast your spices for enhanced flavor", "Use fresh lime juice for brightness"],
            'asian': ["Have all ingredients ready - Asian cooking is fast!", "Use high heat for stir-frying"],
            'indian': ["Bloom spices in oil to release their flavors", "Balance heat with dairy or coconut"],
            'french': ["Cook with patience - French cooking rewards technique", "Use butter for richness"],
            'american': ["Don't flip meat too often while cooking", "Let meat rest after cooking"],
            'mediterranean': ["Use good quality olive oil", "Fresh herbs make a big difference"]
        }
        
        tips = general_tips[:2]  # Start with 2 general tips
        
        # Add cuisine-specific tips
        for cuisine_key, cuisine_specific_tips in cuisine_tips.items():
            if cuisine_key in cuisine:
                tips.extend(cuisine_specific_tips[:2])
                break
        
        return tips[:4]  # Return max 4 tips
    
    def suggest_substitutions(self, recipe):
        """Suggest ingredient substitutions"""
        ingredients = recipe.get('ingredients', [])
        substitutions = {}
        
        common_substitutions = {
            'butter': ['olive oil', 'coconut oil', 'applesauce (for baking)'],
            'milk': ['almond milk', 'oat milk', 'soy milk'],
            'eggs': ['flax eggs', 'applesauce', 'aquafaba'],
            'cheese': ['nutritional yeast', 'cashew cheese', 'vegan cheese'],
            'flour': ['almond flour', 'coconut flour', 'rice flour'],
            'sugar': ['honey', 'maple syrup', 'stevia'],
            'cream': ['coconut cream', 'cashew cream', 'silken tofu']
        }
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            for sub_key, alternatives in common_substitutions.items():
                if sub_key in ingredient_lower:
                    substitutions[ingredient] = alternatives[:3]  # Max 3 alternatives
                    break
        
        return substitutions
