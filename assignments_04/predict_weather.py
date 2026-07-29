'''
Part 2: Mini-Project — Build the Weather Classifier
File 2: predict_weather.py
This script simulates how the trained model would be used in production. It should have no training code — no fit, no GridSearchCV, no raw data loading from the API.
'''
import joblib
import json
import pandas as pd

# ==========================================
# --- Task 1: Load and Verify ---
# ==========================================
print("\n-----Task 1: Load and Verify-----\n")
# Load the Pipeline from models/weather_classifier.pkl. Load the metadata from models/weather_classifier_metadata.json and print the model's key metadata (city, features, test AUC).

# Load the Pipeline
clf = joblib.load("models/weather_classifier.pkl")

# Load the metadata
with open("models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

# Print the model's key metadata
print("Model's Key Metadata:")
print(f"  City: {metadata['location']['city']}")
print(f"  Features: {metadata['features']}")
print(f"  Test AUC: {metadata['test_auc']}")


# ==========================================
# --- Task 2: Predict on New Data ---
# ==========================================
print("\n-----Task 2: Predict on New Data-----\n")
# Create a DataFrame of at least five hypothetical days, covering a range of conditions: clearly good days, clearly bad days, and at least one borderline case. 
# Use the feature names from your metadata to make sure the columns match exactly.

# "label_thresholds": "Good for running: Max temp 7-32C, Min temp >= 0C, Precip < 3.0mm, Wind < 30km/h"

new_days = pd.DataFrame({
    "temperature_2m_max": [22.0, 5.0, 30.0, 14.0, 8.0],
    "temperature_2m_min": [12.0, -3.0, 20.0, 8.0, 0.0],
    "precipitation_sum":  [0.5,   0.0,  8.0, 1.0, 3.0],
    "wind_speed_10m_max": [15.0,  10.0, 45.0, 20.0, 29.0],
})

# For each day, print:

# The four input feature values
# The predicted label (good / skip)
# The model's confidence (probability of "good for running")

# Predict — the pipeline applies scaling automatically
predictions = clf.predict(new_days)
probabilities = clf.predict_proba(new_days)[:, 1]

# Loop through each day and print the results
for i in range(len(new_days)):
    # The four input feature values
    inputs = new_days.iloc[i].to_dict()
    label = "Good" if predictions[i] == 1 else "Skip"
    confidence = probabilities[i] * 100
    
    print(f"\nDay {i+1}")
    print(f"Inputs:     {inputs}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")

# ==========================================
# --- Task 3: Reflect ---
# ==========================================
print("\n-----Task 3: Reflect-----\n")

# Add a comment block answering the following:
# Pick the borderline case you included. What was the probability? Would you describe the model's answer as confident or uncertain? How would you handle a day where the model says 0.52?
# The training script and the prediction script are completely separate. What would break if someone ran predict_weather.py before train_weather_classifier.py? How would you make the error message more helpful?
# In a production system, the prediction script might run daily to classify tomorrow's weather forecast. What would need to change in predict_weather.py to support that? (You do not need to implement this — just describe it in a comment.)

# COMMENT: The borderline case i included was Day 5. The model's confidence (probability) was 70%. I would say the model's answer was pretty confident. I would handle a day where it was 0.52 confident as not good for running. I'd rather not run on a good day than run in a bad day.
# if someone ran predict_weather.py before train_weather_classifier.py, then it wouldn't work because train_weather_classifier.py creates the models that are needed for predict_weather.py. I would make the error message more helpful by letting the user know that you need to run the train_weather_classifier.py first so you can have the prediction model.
# I would need to change the task 2 in predict_weather.py to support actual data. I might need to implement past weather conditions rather than hard-coding new_days.