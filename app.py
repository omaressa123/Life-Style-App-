
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from workout_model import FitnessRecommender

app = Flask(__name__)

# Global variables for models and data
workout_recommender = None
health_recommender_func = None
meal_scaler = None
meal_similarity_matrix = None
meal_df = None
df_workout_data = None # This will be the full dataframe for workout recommendations

def load_models_and_data():
    global workout_recommender, health_recommender_func, meal_scaler, meal_similarity_matrix, meal_df, df_workout_data
    if workout_recommender is None:
        try:
            print("Attempting to load Final_data.csv...")
            df_workout_data = pd.read_csv("Final_data.csv")
            print(f"Final_data.csv loaded. Columns: {df_workout_data.columns.tolist()}")
            print("Attempting to load fitness_recommender.pkl...")
            workout_recommender = joblib.load("fitness_recommender.pkl")
            print("Workout Recommender loaded successfully.")
        except FileNotFoundError:
            workout_recommender = None
            print(f"Error: Final_data.csv or fitness_recommender.pkl not found for workout recommender.")
        except Exception as e:
            workout_recommender = None
            print(f"An unexpected error occurred loading workout recommender: {e}")

    if health_recommender_func is None:
        try:
            # The health system doesn't use a saved class, but the function itself.
            # So we'll just use the function directly.
            health_recommender_func = health_recommendation_system
            print("Health & Lifestyle Recommender loaded successfully.")
        except Exception as e:
            health_recommender_func = None
            print(f"Error loading health & lifestyle recommender: {e}")

    if meal_df is None or meal_scaler is None or meal_similarity_matrix is None:
        try:
            meal_scaler = joblib.load("scaler_meal_recommender.pkl")
            meal_similarity_matrix = joblib.load("meal_similarity_matrix.pkl")
            meal_df = pd.read_csv("meal_metadata.csv")
            print("Meal Recommender loaded successfully.")
        except Exception as e:
            meal_scaler = None
            meal_similarity_matrix = None
            meal_df = None
            print(f"Error loading meal recommender: {e}")


def health_recommendation_system(user):
    recommendations = []
    score = 0  # out of 10
    
    # 🎯 BMI Analysis
    if user['BMI'] < 18.5:
        recommendations.append(" Increase calorie intake with balanced protein meals (underweight).")
        score += 6
    elif user['BMI'] <= 24.9:
        recommendations.append(" Your BMI is in a healthy range — maintain your current diet and activity.")
        score += 9
    elif user['BMI'] >= 30.0:
        recommendations.append(" Slightly overweight — focus on moderate cardio and portion control.")
        score += 7
    else:
        recommendations.append(" High BMI — increase cardio sessions and reduce high-fat/sugar meals.")
        score += 5
    
    # 💧 Water Intake
    if user['Water_Intake (liters)'] < 1.5:
        recommendations.append(" Increase water intake to at least 2.5 liters per day.")
        score += 5
    elif user['Water_Intake (liters)'] < 2.5:
        recommendations.append("Drink slightly more water, aim for 2.5L daily.")
        score += 7
    else:
        recommendations.append(" Excellent hydration habits!")
        score += 9
    
    # 🏋️ Workout Frequency
    if user['Workout_Frequency (days/week)'] < 2:
        recommendations.append(" Start exercising at least 3 times a week.")
        score += 5
    elif user['Workout_Frequency (days/week)'] < 4:
        recommendations.append(" Good activity level, try to increase intensity gradually.")
        score += 8
    else:
        recommendations.append(" Great consistency! Maintain your routine.")
        score += 10

    # 🧠 Physical Exercise Type
    if "cardio" in str(user['Physical exercise']).lower():
        recommendations.append(" Excellent — cardio improves heart and stamina.")
        score += 9
    elif "strength" in str(user['Physical exercise']).lower():
        recommendations.append(" Strength training builds long-term metabolism — keep it up!")
        score += 9
    else:
        recommendations.append(" Add variety — mix cardio, flexibility, and strength workouts.")
        score += 7

    # 🍱 Meal Frequency
    meals = user['Daily meals frequency']
    if meals < 3:
        recommendations.append(" Increase to 3–5 balanced meals daily to stabilize metabolism.")
        score += 6
    elif meals <= 5:
        recommendations.append(" Perfect meal frequency — keep meals balanced with proteins and veggies.")
        score += 9
    else:
        recommendations.append(" Too frequent meals — may increase caloric load, try to space them out.")
        score += 6

    # 🧾 Final Lifestyle Score
    lifestyle_score = round(score / 5, 2)
    recommendations.append(f" Estimated Lifestyle Score: {lifestyle_score}/10")

    # 🧭 Summary Recommendation
    if lifestyle_score >= 8:
        recommendations.append(" You’re maintaining a healthy lifestyle! Continue consistency.")
    elif lifestyle_score >= 6:
        recommendations.append("Good progress! Focus on minor improvements in hydration or exercise.")
    else:
        recommendations.append(" Needs attention — adjust nutrition, water, and activity levels.")

    return recommendations ,lifestyle_score

def recommend_meal(meal_name, top_n=5):
    if meal_name not in meal_df['meal_name'].values:
        return f"Meal '{meal_name}' not found. Try another one."
    
    idx = meal_df.index[meal_df['meal_name'] == meal_name][0]
    sim_scores = list(enumerate(meal_similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_meals = [i[0] for i in sim_scores[1:top_n+1]]
    return meal_df.iloc[top_meals][['meal_name', 'meal_type', 'diet_type', 'Calories', 'Proteins', 'Carbs', 'Fats', 'cooking_method']]


@app.route('/', methods=['GET', 'POST'])
def home():
    load_models_and_data() # Ensure models and data are loaded

    workout_recommendations = None
    health_recommendations = None
    meal_recommendations = None

    # Populate dropdown options
    genders = ['Male', 'Female']
    experience_levels = ['Beginner', 'Intermediate', 'Advanced'] # Mapped from numerical
    target_muscle_groups = sorted(list(df_workout_data['Target Muscle Group'].unique())) if df_workout_data is not None else []
    difficulty_levels = sorted(list(df_workout_data['Difficulty Level'].unique())) if df_workout_data is not None else []
    physical_exercises = ['Cardio', 'Strength', 'Yoga', 'HIIT', 'Stretching', 'Other']
    
    meal_names = sorted(list(meal_df['meal_name'].unique())) if meal_df is not None else []
    meal_types = sorted(list(meal_df['meal_type'].unique())) if meal_df is not None else []
    diet_types = sorted(list(meal_df['diet_type'].unique())) if meal_df is not None else []
    cooking_methods = sorted(list(meal_df['cooking_method'].unique())) if meal_df is not None else []


    if request.method == 'POST':
        if 'workout_submit' in request.form:
            difficulty = request.form.get('workout_difficulty')
            muscle_group = request.form.get('workout_muscle_group')
            n = int(request.form.get('workout_n', 3))

            if workout_recommender and difficulty:
                try:
                    workout_recommendations = workout_recommender.recommend(difficulty, muscle_group, n).to_dict(orient='records')
                except Exception as e:
                    workout_recommendations = [{"error": str(e)}]
            else:
                workout_recommendations = [{"error": "Workout recommender not loaded or difficulty missing."}]

        elif 'health_submit' in request.form:
            user_data = {
                'Age': int(request.form.get('health_age')),
                'Gender': request.form.get('health_gender'),
                'BMI': float(request.form.get('health_bmi')),
                'Fat_Percentage': float(request.form.get('health_fat_percentage')),
                'Workout_Frequency (days/week)': float(request.form.get('health_workout_frequency')),
                'Physical exercise': request.form.get('health_physical_exercise'),
                'Water_Intake (liters)': float(request.form.get('health_water_intake')),
                'Daily meals frequency': float(request.form.get('health_daily_meals_frequency'))
            }
            if health_recommender_func:
                try:
                    recommendations, lifestyle_score = health_recommender_func(user_data)
                    health_recommendations = {"recommendations": recommendations, "lifestyle_score": lifestyle_score}
                except Exception as e:
                    health_recommendations = {"error": str(e)}
            else:
                health_recommendations = {"error": "Health recommender not loaded."}

        elif 'meal_submit' in request.form:
            meal_name = request.form.get('meal_meal_name')
            top_n = int(request.form.get('meal_top_n', 2))

            if meal_df is not None and meal_scaler is not None and meal_similarity_matrix is not None and meal_name:
                try:
                    recs = recommend_meal(meal_name, top_n)
                    if isinstance(recs, str):
                        meal_recommendations = [{"error": recs}]
                    else:
                        meal_recommendations = recs.to_dict(orient='records')
                except Exception as e:
                    meal_recommendations = [{"error": str(e)}]
            else:
                meal_recommendations = [{"error": "Meal recommender not loaded or meal name missing."}]

    return render_template('index.html',
                           genders=genders,
                           experience_levels=experience_levels,
                           target_muscle_groups=target_muscle_groups,
                           difficulty_levels=difficulty_levels,
                           physical_exercises=physical_exercises,
                           meal_names=meal_names,
                           meal_types=meal_types,
                           diet_types=diet_types,
                           cooking_methods=cooking_methods,
                           workout_recommendations=workout_recommendations,
                           health_recommendations=health_recommendations,
                           meal_recommendations=meal_recommendations)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
