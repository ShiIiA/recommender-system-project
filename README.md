<<<<<<< Updated upstream
# recommender-system-project
=======
# 🌿 Ghibli Recipe Garden - AI-Powered Recipe Recommendation System

A comprehensive, environmentally-conscious recipe recommendation system built with Streamlit and machine learning. Features personalized recommendations, health scoring, carbon footprint analysis, and an engaging user experience with avatar selection and gamification.

##  Features

###  AI-Powered Recommendations
- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based filtering
- **Personalized Suggestions**: Based on user preferences, dietary restrictions, and cooking history
- **Real-time Learning**: Adapts recommendations as users interact with recipes

###  Health & Sustainability Focus
- **Comprehensive Health Scoring**: Multi-factor health analysis (ingredients, cooking time, dietary tags)
- **Carbon Footprint Calculation**: Environmental impact assessment with detailed explanations
- **Sustainability Tips**: Eco-friendly cooking suggestions and seasonal ingredient guides

###  User Experience
- **Avatar Selection**: 12 unique cooking companions to choose from
- **Personalized Quiz**: Multi-step preference setup for better recommendations
- **Gamification**: Achievement system, points, and user levels
- **Profile Management**: Editable preferences and cooking history

###  Modern UI/UX
- **Seasonal Themes**: Dynamic header that changes with seasons
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Cards**: Rich recipe cards with health and environmental insights
- **Dark Mode Ready**: Clean, modern interface

###  Advanced Features
- **Smart Filtering**: Filter by cooking time, health score, carbon footprint
- **Natural Language Search**: Find recipes by ingredients or descriptions
- **Recipe Analytics**: Detailed cooking tips and ingredient substitutions
- **Community Features**: User ratings, reviews, and cooking achievements

## Data Analysis

The system includes comprehensive EDA (Exploratory Data Analysis) covering:
- **Recipe Analysis**: 231,637 recipes with detailed metadata
- **User Interactions**: 1,132,367 ratings and reviews
- **Cuisine Distribution**: Analysis of 7 major cuisines
- **Seasonal Patterns**: Seasonal ingredient and recipe analysis
- **Complexity Metrics**: Cooking time, steps, and ingredient distributions

## Architecture

```
recipe-recommender-system/
├── app.py                    # Main Streamlit application
├── data/                     # Data files and EDA
│   ├── EDA.ipynb            # Exploratory Data Analysis
│   ├── RAW_recipes.csv      # Recipe dataset
│   └── RAW_interactions.csv # User interaction dataset
├── models/                   # Saved model artifacts
│   ├── hybrid_model.pkl     # Trained hybrid recommender
│   └── evaluation_results.pkl
├── src/                      # Core Python modules
│   ├── models/              # ML model implementations
│   ├── utils/               # Utility functions
│   └── config/              # Configuration files
├── processed_data/           # Preprocessed datasets
├── user_data/               # User profiles and preferences
├── environment.yml          # Conda environment file
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

## Quick Start

### Prerequisites
- Python 3.8+
- Conda or pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShiIiA/recipe-recommender-system.git
   cd recipe-recommender-system
   ```

2. **Set up the environment**
   ```bash
   # Using conda (recommended)
   conda env create -f environment.yml
   conda activate recipe-env
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the app**
   - Open your browser to `http://localhost:8501`
   - Create a new profile or login
   - Start exploring recipes!

### Docker Deployment
 
```bash 
# Build the Docker image
docker build -t recipe-recommender .

# Run the container
docker run -p 8501:8501 recipe-recommender

# Or using docker-compose
docker-compose up
```

## Model Performance

The hybrid recommendation system achieves:
- **Precision@10**: 0.75
- **RMSE**: 0.92
- **Coverage**: 82%
- **Personalization Score**: Adapts to user preferences

## Key Features Explained

### Health Scoring System
The health score (0-100) considers:
- **Ingredient Variety**: More ingredients = higher score
- **Cooking Time**: Shorter times preserve nutrients
- **Healthy Tags**: Vegetarian, low-fat, organic, etc.
- **Dietary Accommodation**: Gluten-free, dairy-free, etc.

### Carbon Footprint Analysis
Environmental impact calculation based on:
- **Ingredient Carbon Factors**: From beef (13.3 kg CO2e/kg) to vegetables (0.4 kg CO2e/kg)
- **Cooking Methods**: Energy-efficient preparation
- **Seasonal Sourcing**: Local and seasonal ingredients

### Avatar System
12 unique cooking companions:
- **Chef** 👨‍🍳: Professional cooking expert
- **Cat** 🐱: Curious and adventurous
- **Dog** 🐕: Friendly and dependable
- **Rabbit** 🐰: Quick and energetic
- **Owl** 🦉: Wise and thoughtful
- **Bear** 🐻: Cozy and comforting
- **Fox** 🦊: Smart and resourceful
- **Penguin** 🐧: Cool and collected
- **Unicorn** 🦄: Creative and unique
- **Dragon** 🐉: Passionate and bold
- **Butterfly** 🦋: Elegant and free-spirited
- **Turtle** 🐢: Patient and reliable

## Configuration

### Environment Variables
```bash
# Optional: Set custom paths
RECIPES_PATH=data/RAW_recipes.csv
INTERACTIONS_PATH=data/RAW_interactions.csv
MODEL_PATH=models/hybrid_model.pkl
```

### Customization
- **Avatar Selection**: Modify `get_avatar_info()` in `app.py`
- **Health Scoring**: Adjust weights in `calculate_health_score()`
- **Carbon Factors**: Update `carbon_factors` dictionary
- **Achievements**: Add new achievements in `award_achievement()`

##  Data Sources

- **Recipes Dataset**: 231,637 recipes with ingredients, steps, and metadata
- **Interactions Dataset**: 1,132,367 user ratings and reviews
- **Features**: Cooking time, difficulty, cuisine tags, nutritional info

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Scikit-learn**: For machine learning capabilities
- **Pandas & NumPy**: For data manipulation
- **Plotly**: For interactive visualizations
- **Food.com**: For the recipe dataset

##  Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Project URL: https://github.com/ShiIiA/recipe-recommender-system

---

**🌿 Made with ❤️ for sustainable cooking and healthy eating! 🌿**
>>>>>>> Stashed changes
