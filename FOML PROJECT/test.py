from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- STEP 1: Load YOLO Model and Calorie Data ---
model = YOLO("runs/detect/BurnGain/weights/best.pt")
cal_df = pd.read_csv("calories.csv")
cal_df['Food'] = cal_df['Food'].str.lower().str.strip()
cal_df = cal_df.drop_duplicates(subset='Food')
cal_dict = cal_df.set_index('Food')[['Grams', 'Calories']].to_dict('index')

# --- STEP 2: Detect food and estimate total calories ---
img_path = "t1.jpg"
results = model(img_path, conf=0.01)[0]

total_cals = 0
print("\n Detected Foods and Calorie Estimates:")
for box in results.boxes:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    food_key = cls_name.lower().strip()
    xywh = box.xywh[0]
    w, h = xywh[2].item(), xywh[3].item()
    area = w * h
    grams = min((area / 50000) * 100, 1000)

    print(f"Food: {cls_name}")
    print(f"Estimated Grams: {grams:.1f}g")

    if food_key in cal_dict:
        base_grams = cal_dict[food_key]['Grams']
        base_cals = cal_dict[food_key]['Calories']
        cals = (grams / base_grams) * base_cals
        total_cals += cals
        print(f"Estimated Calories: {cals:.1f} kcal\n")
    else:
        print("Calories: Not available\n")

print(f"\nTotal Calories to Burn: {total_cals:.2f} kcal")

# --- STEP 3: Load and train exercise model ---
exercise_df = pd.read_csv("synthetic_exercise_dataset.csv")
le = LabelEncoder()
exercise_df['Gender'] = le.fit_transform(exercise_df['Gender'])

# --- STEP 4: Get user input ---
weight = float(input("Enter your weight (kg): "))
age = int(input("Enter your age: "))
gender = input("Enter your gender (Male/Female): ").strip().capitalize()
height = float(input("Enter your height (cm): "))
time_limit = float(input("Maximum time you can exercise (in minutes): "))

gender_encoded = 1 if gender == "Male" else 0
height_m = height / 100
bmi = weight / (height_m ** 2)

# Add user BMI to all rows (since dataset has no height column)
exercise_df['BMI'] = bmi

# Prepare training data
features = ['Weight', 'Age', 'Gender', 'Duration', 'BMI']
target = 'Calories Burn'

X = exercise_df[features]
y = exercise_df[target]

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X, y)

# --- STEP 5: Predict burn for all exercises ---
exercise_df['Predicted Burn'] = model_rf.predict(X)

# Filter by user constraints
user_matches = exercise_df[ 
    (exercise_df['Weight'] == weight) & 
    (exercise_df['Age'] == age) & 
    (exercise_df['Gender'] == gender_encoded)
]

if user_matches.empty:
    # Fallback: Predict for each exercise with user's profile
    exercise_options = []
    for ex in exercise_df['Exercise'].unique():
        durations = exercise_df[exercise_df['Exercise'] == ex]['Duration'].unique()
        for dur in durations:
            row = pd.DataFrame({
                'Weight': [weight],
                'Age': [age],
                'Gender': [gender_encoded],
                'Duration': [dur],
                'BMI': [bmi]
            })
            pred = model_rf.predict(row)[0]
            if pred >= total_cals and dur <= time_limit:
                exercise_options.append((ex, pred, dur))

    exercise_options = sorted(exercise_options, key=lambda x: x[2])
    if exercise_options:
        print(f"\nTop 5 Exercises that can burn {total_cals:.1f} kcal within {time_limit} minutes:")
        print(f"{'Exercise':<15}{'Calories Burn':<18}{'Duration (mins)'}")
        
        # Sort the options by predicted burn and select top 5
        top_5_exercises = sorted(exercise_options, key=lambda x: x[1], reverse=True)[:5]
        
        for ex, cal, dur in top_5_exercises:
            print(f"{ex:<15}{cal:<18.2f}{dur}")
        
        # Find the fastest time to burn the target calories
        best_exercise = top_5_exercises[0]
        best_exercise_name, best_exercise_calories, best_exercise_duration = best_exercise

        # Calculate the time needed to burn target calories
        burn_per_minute = best_exercise_calories / best_exercise_duration
        time_to_burn_target = total_cals / burn_per_minute

        print(f"\nFastest Option: {best_exercise_name} in {best_exercise_duration} minutes")
        print(f"It will take {time_to_burn_target:.2f} minutes to burn {total_cals:.1f} kcal with {best_exercise_name}.")
    else:
        print("\nNo exercises found that meet the goal within the time limit. Try increasing time.")
else:
    valid_ex = user_matches[
        (user_matches['Predicted Burn'] >= total_cals) &
        (user_matches['Duration'] <= time_limit)
    ].sort_values(by='Duration')

    if not valid_ex.empty:
        print(f"\nTop 5 Exercises that can burn {total_cals:.1f} kcal within {time_limit} minutes:")
        print(f"{'Exercise':<15}{'Calories Burn':<18}{'Duration (mins)'}")
        
        # Sort the exercises by predicted burn and select the top 5
        top_5_valid_exercises = valid_ex.sort_values(by='Predicted Burn', ascending=False).head(5)
        
        for _, row in top_5_valid_exercises.iterrows():
            print(f"{row['Exercise']:<15}{row['Predicted Burn']:<18.2f}{row['Duration']}")
        
        # Find the best exercise and calculate the time needed
        best_ex = top_5_valid_exercises.iloc[0]
        best_ex_name = best_ex['Exercise']
        best_ex_calories = best_ex['Predicted Burn']
        best_ex_duration = best_ex['Duration']

        # Calculate time to burn target calories
        burn_per_minute = best_ex_calories / best_ex_duration
        time_to_burn_target = total_cals / burn_per_minute

        print(f"\nFastest Option: {best_ex_name} in {best_ex_duration} minutes")
        print(f"It will take {time_to_burn_target:.2f} minutes to burn {total_cals:.1f} kcal with {best_ex_name}.")
    else:
        print("\nNo suitable exercises found for your profile within the time.")
