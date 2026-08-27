'''
Part 2: Mini-Project — Build the Weather Classifier
File 1: train_weather_classifier.py
This script does all the heavy lifting: data loading, label engineering, model selection, final evaluation, and saving.
'''

import requests
import pandas as pd
import joblib
import json
import sklearn
import sys
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    )

# ==========================================
# --- Step 1: Fetch the Data ---
# ==========================================
print("\n-----Step 1: Fetch the Data-----\n")

# Use the Open-Meteo historical API to download one year of daily weather data for a location of your choice. The API is free and requires no key. Use these four daily variables:

import pandas as pd
import requests

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 43.7314,
	"longitude": 7.419,
	"start_date": "2025-06-01",
	"end_date": "2026-06-01",
	"daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
	"timezone": "auto",
}
response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

# Print a summary of the dataset
print("\nData Summary Monaco 06/01/2025 - 06/01/2026:")
print(df.info())
print("\n")
print(df.describe())


# ==========================================
# --- Step 2: Engineer Labels ---
# ==========================================
print("\n-----Step 2: Engineer Labels-----\n")

def label_running_day(row):
    return int(
        7 <= row["temperature_2m_max"] <= 32  #(45-90°F) I can run in 90 degree weather Farenheight 
        and row["temperature_2m_min"] >= 0
        and row["precipitation_sum"] < 3.0
        and row["wind_speed_10m_max"] < 30
    )

df['good_for_running'] = df.apply(label_running_day,axis=1)

# Print the class distribution (how many days are 1 vs 0)
print("Class Distribution (Counts):")
print(df['good_for_running'].value_counts())

# Calculate and print the fraction of good days
fraction_good = df['good_for_running'].mean()
print(f"\nFraction of good running days: {fraction_good:.2%}")

# add a comment: what fraction of days in your dataset are labeled "good for running"? Does that seem reasonable given the climate where you chose?
# COMMENT: About 80% of the days in Monaco from my dataset is 'good for running' for me. It's reasonable since I know monaco doesn't have crazy weather over there.

# ==========================================
# --- Step 3: Train and Tune ---
# ==========================================
print("\n-----Step 3: Train and Tune-----\n")

# Split the data into train (80%) and test (20%) sets, stratifying on the label.
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use GridSearchCV with a Pipeline(StandardScaler, LogisticRegression) to search over at least five values of C. Use cv=5 and scoring="roc_auc". 

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train,y_train)

# Print:
# The best C value and best CV AUC
# A full classification report on the test set
# The test AUC

best_pipe = grid_search.best_estimator_
y_probs = best_pipe.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_probs)
y_pred = best_pipe.predict(X_test)
print(f"Best C: {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
print(f"Full Classification Report on Test Set:\n{classification_report(y_true=y_test,y_pred=y_pred)}")
print(f"Test AUC: {test_auc:.3f}")

# Then plot and save the ROC curve for the best estimator to outputs/weather_roc.png.
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name="Logistic Regression")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve — Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.close()

# ==========================================
# --- Step 4: Reflect on Evaluation ---
# ==========================================
print("\n-----Step 4: Reflect on Evaluation-----\n")

# Add a comment block (at least 4-6 sentences) addressing the following:

# What does the AUC score tell you about this model's quality? Is it surprisingly good, surprisingly bad, or about what you expected?
# Look at the precision and recall in the classification report. Which type of error (false positive vs. false negative) is more common? What would this mean in practice — would you rather the app over-recommend running or under-recommend it?
# If you were setting the threshold for a real app, would you use the default 0.5? What would you change it to and why?

# COMMENT: The AUC score is 0.982, meaning the model's doing a good job separating good days and bad days. I'd say it's about what I'd expect, especially with Monaco not being known for having extreme weather.
# Looking at the report, false positives are way more common. It caught 100% of the good days but accidentally labeled a few bad days as good ones. In real life, this means the app would over-recommend running. I'd honestly rather it under-recommend—getting caught outside in bad weather sucks
# Because of that, I wouldn't use the default 0.5 threshold. I'd raise it to something higher, like 0.7 or 0.8, so the model has to be way more confident that it's actually nice out before telling me to go for a run.


# ==========================================
# --- Step 5: Save the Model ---
# ==========================================
print("\n-----Step 5: Save the Model-----\n")

# Save the best Pipeline (scaler + logistic regression)
joblib.dump(best_pipe, "models/weather_classifier.pkl")

# metadata into dictionary
metadata = {
    "python_version": sys.version,
    "scikit_learn_version": sklearn.__version__,
    "features": FEATURES,
    "best_hyperparameters": grid_search.best_params_,
    "test_auc": round(test_auc, 3),
    "location": {
        "city": "Monaco",
        "latitude": 43.7314,
        "longitude": 7.419
    },
    "label_thresholds": "Good for running: Max temp 7-32C, Min temp >= 0C, Precip < 3.0mm, Wind < 30km/h"
}

# JSON
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Print confirmation message
print("Success! Model saved to 'models/weather_classifier.pkl'")
print("Success! Metadata saved to 'models/weather_classifier_metadata.json'\n")

