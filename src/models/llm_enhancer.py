"""
LLM Enhancer for recipe recommendations and descriptions
"""
import random
from typing import Dict, List, Any

class LocalLLMEnhancer:
    """Local LLM enhancer for recipe recommendations and descriptions"""
    
    def __init__(self):
        self.enhancement_templates = {
            'descriptions': [
                "A delightful {cuisine} dish featuring {main_ingredient} that's perfect for {occasion}.",
                "This {style} recipe combines {ingredients} for a {flavor} experience.",
                "Experience the {region} flavors with this {cooking_method} {dish_type}.",
                "A {health_benefit} {meal_type} that's both {taste} and {texture}.",
                "Perfect for {skill_level} cooks, this {time} recipe delivers {outcome}."
            ],
            'cooking_tips': [
                "For best results, ensure all ingredients are at room temperature.",
                "Don't rush the cooking process - patience leads to better flavors.",
                "Taste as you go and adjust seasoning to your preference.",
                "Use fresh ingredients whenever possible for the best outcome.",
                "Let the dish rest for a few minutes before serving for optimal flavor."
            ],
            'substitutions': {
                'butter': ['olive oil', 'coconut oil', 'avocado oil'],
                'milk': ['almond milk', 'soy milk', 'oat milk'],
                'eggs': ['flax eggs', 'chia eggs', 'banana'],
                'sugar': ['honey', 'maple syrup', 'stevia'],
                'flour': ['almond flour', 'coconut flour', 'oat flour']
            }
        }
    
    def generate_recipe_description(self, recipe: Dict[str, Any]) -> str:
        """Generate an enhanced recipe description"""
        try:
            # Extract recipe features
            ingredients = recipe.get('ingredients', [])
            tags = recipe.get('tags', [])
            minutes = recipe.get('minutes', 30)
            
            # Determine cuisine type from tags
            cuisine = self._extract_cuisine(tags)
            main_ingredient = self._get_main_ingredient(ingredients)
            occasion = self._get_occasion(tags, minutes)
            style = self._get_cooking_style(tags)
            flavor = self._get_flavor_profile(tags)
            region = self._get_region(cuisine)
            cooking_method = self._get_cooking_method(tags)
            dish_type = self._get_dish_type(tags)
            health_benefit = self._get_health_benefit(tags)
            meal_type = self._get_meal_type(tags)
            taste = self._get_taste(tags)
            texture = self._get_texture(tags)
            skill_level = self._get_skill_level(minutes, len(ingredients))
            time = self._get_time_description(minutes)
            outcome = self._get_outcome(tags)
            
            # Select and fill template
            template = random.choice(self.enhancement_templates['descriptions'])
            description = template.format(
                cuisine=cuisine,
                main_ingredient=main_ingredient,
                occasion=occasion,
                style=style,
                ingredients=self._format_ingredients(ingredients[:3]),
                flavor=flavor,
                region=region,
                cooking_method=cooking_method,
                dish_type=dish_type,
                health_benefit=health_benefit,
                meal_type=meal_type,
                taste=taste,
                texture=texture,
                skill_level=skill_level,
                time=time,
                outcome=outcome
            )
            
            return description
            
        except Exception as e:
            # Fallback to original description
            return recipe.get('description', "A delicious recipe worth trying!")
    
    def enhance_recommendations(self, recipes: List[Dict], user_profile: Dict) -> List[Dict]:
        """Enhance recipe recommendations with personalization"""
        enhanced_recipes = []
        
        for recipe in recipes:
            enhanced_recipe = recipe.copy()
            
            # Add personalization score based on user preferences
            personalization_score = self._calculate_personalization_score(recipe, user_profile)
            enhanced_recipe['personalization_score'] = personalization_score
            
            # Add enhanced description
            enhanced_recipe['enhanced_description'] = self.generate_recipe_description(recipe)
            
            enhanced_recipes.append(enhanced_recipe)
        
        return enhanced_recipes
    
    def generate_cooking_tips(self, recipe: Dict[str, Any]) -> List[str]:
        """Generate cooking tips for a recipe"""
        tips = []
        
        # Add general tips
        tips.extend(random.sample(self.enhancement_templates['cooking_tips'], 2))
        
        # Add specific tips based on recipe features
        ingredients = recipe.get('ingredients', [])
        tags = recipe.get('tags', [])
        minutes = recipe.get('minutes', 30)
        
        if any('fish' in ing.lower() for ing in ingredients):
            tips.append("Cook fish until it flakes easily with a fork for perfect doneness.")
        
        if any('pasta' in ing.lower() for ing in ingredients):
            tips.append("Reserve some pasta water to adjust sauce consistency if needed.")
        
        if minutes > 60:
            tips.append("This recipe benefits from slow cooking - don't rush the process.")
        
        if len(ingredients) > 10:
            tips.append("Prep all ingredients before starting to make cooking smoother.")
        
        return tips[:3]  # Return top 3 tips
    
    def suggest_substitutions(self, recipe: Dict[str, Any]) -> Dict[str, List[str]]:
        """Suggest ingredient substitutions"""
        ingredients = recipe.get('ingredients', [])
        substitutions = {}
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            for original, alternatives in self.enhancement_templates['substitutions'].items():
                if original in ingredient_lower:
                    substitutions[ingredient] = alternatives
                    break
        
        return substitutions
    
    def _extract_cuisine(self, tags: List[str]) -> str:
        """Extract cuisine type from tags"""
        cuisines = ['italian', 'mexican', 'chinese', 'indian', 'french', 'japanese', 'thai', 'mediterranean']
        for tag in tags:
            for cuisine in cuisines:
                if cuisine in tag.lower():
                    return cuisine.title()
        return "International"
    
    def _get_main_ingredient(self, ingredients: List[str]) -> str:
        """Get main ingredient from recipe"""
        if not ingredients:
            return "fresh ingredients"
        
        # Look for protein or main vegetables
        main_ingredients = ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'tofu', 'pasta', 'rice']
        for ingredient in ingredients:
            for main in main_ingredients:
                if main in ingredient.lower():
                    return ingredient
        return ingredients[0]
    
    def _get_occasion(self, tags: List[str], minutes: int) -> str:
        """Get suitable occasion for the recipe"""
        if minutes < 30:
            return "quick weeknight dinners"
        elif minutes < 60:
            return "weekend cooking"
        else:
            return "special occasions"
    
    def _get_cooking_style(self, tags: List[str]) -> str:
        """Get cooking style from tags"""
        styles = ['traditional', 'modern', 'fusion', 'classic', 'contemporary']
        for tag in tags:
            for style in styles:
                if style in tag.lower():
                    return style
        return "delicious"
    
    def _get_flavor_profile(self, tags: List[str]) -> str:
        """Get flavor profile from tags"""
        flavors = ['spicy', 'sweet', 'savory', 'tangy', 'herbaceous', 'rich']
        for tag in tags:
            for flavor in flavors:
                if flavor in tag.lower():
                    return flavor
        return "delicious"
    
    def _get_region(self, cuisine: str) -> str:
        """Get region from cuisine"""
        region_map = {
            'Italian': 'Mediterranean',
            'Mexican': 'Latin American',
            'Chinese': 'Asian',
            'Indian': 'South Asian',
            'French': 'European',
            'Japanese': 'East Asian',
            'Thai': 'Southeast Asian'
        }
        return region_map.get(cuisine, 'International')
    
    def _get_cooking_method(self, tags: List[str]) -> str:
        """Get cooking method from tags"""
        methods = ['baked', 'grilled', 'fried', 'steamed', 'roasted', 'braised']
        for tag in tags:
            for method in methods:
                if method in tag.lower():
                    return method
        return "prepared"
    
    def _get_dish_type(self, tags: List[str]) -> str:
        """Get dish type from tags"""
        types = ['soup', 'salad', 'pasta', 'curry', 'stew', 'casserole']
        for tag in tags:
            for dish_type in types:
                if dish_type in tag.lower():
                    return dish_type
        return "dish"
    
    def _get_health_benefit(self, tags: List[str]) -> str:
        """Get health benefit from tags"""
        if any('healthy' in tag.lower() for tag in tags):
            return "nutritious"
        elif any('low-fat' in tag.lower() for tag in tags):
            return "light"
        elif any('vegetarian' in tag.lower() for tag in tags):
            return "plant-based"
        return "wholesome"
    
    def _get_meal_type(self, tags: List[str]) -> str:
        """Get meal type from tags"""
        if any('breakfast' in tag.lower() for tag in tags):
            return "breakfast"
        elif any('lunch' in tag.lower() for tag in tags):
            return "lunch"
        elif any('dinner' in tag.lower() for tag in tags):
            return "dinner"
        return "meal"
    
    def _get_taste(self, tags: List[str]) -> str:
        """Get taste description"""
        if any('spicy' in tag.lower() for tag in tags):
            return "spicy"
        elif any('sweet' in tag.lower() for tag in tags):
            return "sweet"
        return "delicious"
    
    def _get_texture(self, tags: List[str]) -> str:
        """Get texture description"""
        if any('crispy' in tag.lower() for tag in tags):
            return "crispy"
        elif any('creamy' in tag.lower() for tag in tags):
            return "creamy"
        return "satisfying"
    
    def _get_skill_level(self, minutes: int, n_ingredients: int) -> str:
        """Get skill level based on recipe complexity"""
        if minutes < 30 and n_ingredients < 8:
            return "beginner"
        elif minutes < 60 and n_ingredients < 12:
            return "intermediate"
        else:
            return "advanced"
    
    def _get_time_description(self, minutes: int) -> str:
        """Get time description"""
        if minutes < 30:
            return "quick"
        elif minutes < 60:
            return "moderate"
        else:
            return "slow-cooked"
    
    def _get_outcome(self, tags: List[str]) -> str:
        """Get expected outcome"""
        if any('comfort' in tag.lower() for tag in tags):
            return "comforting satisfaction"
        elif any('elegant' in tag.lower() for tag in tags):
            return "elegant dining experience"
        return "delicious results"
    
    def _format_ingredients(self, ingredients: List[str]) -> str:
        """Format ingredients for description"""
        if not ingredients:
            return "fresh ingredients"
        if len(ingredients) == 1:
            return ingredients[0]
        elif len(ingredients) == 2:
            return f"{ingredients[0]} and {ingredients[1]}"
        else:
            return f"{', '.join(ingredients[:-1])}, and {ingredients[-1]}"
    
    def _calculate_personalization_score(self, recipe: Dict, user_profile: Dict) -> float:
        """Calculate personalization score based on user preferences"""
        score = 0.5  # Base score
        
        # Check dietary preferences
        user_diet = user_profile.get('diet', [])
        recipe_tags = recipe.get('tags', [])
        
        for diet in user_diet:
            for tag in recipe_tags:
                if diet.lower() in tag.lower():
                    score += 0.1
        
        # Check cuisine preferences
        user_cuisines = user_profile.get('cuisine_preferences', [])
        for cuisine in user_cuisines:
            for tag in recipe_tags:
                if cuisine.lower() in tag.lower():
                    score += 0.1
        
        # Check cooking time preference
        user_time = user_profile.get('preferred_time', '')
        recipe_time = recipe.get('minutes', 30)
        
        if 'Under 15 minutes' in user_time and recipe_time < 15:
            score += 0.1
        elif '15–30 minutes' in user_time and 15 <= recipe_time <= 30:
            score += 0.1
        elif '30–60 minutes' in user_time and 30 <= recipe_time <= 60:
            score += 0.1
        elif 'Over 60 minutes' in user_time and recipe_time > 60:
            score += 0.1
        
        return min(1.0, max(0.0, score)) 