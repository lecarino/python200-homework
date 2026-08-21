'''
Part 1: Warmup Exercises
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================================
# --- ROC and AUC ---
# ==========================================
print("\n-----ROC and AUC Q1-----\n")

# Train a LogisticRegression(max_iter=1000, random_state=42) on the raw (unscaled) training data and a KNeighborsClassifier(n_neighbors=5) on the scaled training data. For each model:
# Compute predicted probabilities on the test set using .predict_proba()
# Compute and print the AUC score using roc_auc_sco

#Logistic Regression: 
# Create:
lr = LogisticRegression(max_iter=1000, random_state=42)
# Fit:
lr.fit(X_train,y_train)
# Predict: 
y_probs_lr = lr.predict_proba(X_test)[:, 1]
# AUC
auc_lr = roc_auc_score(y_test,y_probs_lr)
# Print
print(f"LR AUC score: {auc_lr:.3f}")

# KNN
# Scale:
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# Create:
knn = KNeighborsClassifier(n_neighbors=5)
# Fit:
knn.fit(X_train_scaled,y_train)
# Predict:
y_probs_knn = knn.predict_proba(X_test_scaled)[:, 1]
# AUC: 
auc_knn = roc_auc_score(y_test,y_probs_knn)
# Print: 
print(f"KNN Scaled AUC score: {auc_knn:.3f}")

# Add a comment: which model has higher AUC? What does that tell you about which model better separates the two classes, independently of any threshold choice?
# COMMENT: The KNN Scaled model has a higher AUC. The KNN Scaled model better separates the two classes because a higher AUC means that, across all possible classification thresholds, 
# the model has a higher probability of correctly ranking a positive example higher than a negative one. 

# ==========================================
# --- ROC Question 2---
# ==========================================
print("\n-----ROC and AUC Q2-----\n")

# Plot both ROC curves on the same axes. Label each curve with the model name and its AUC score. Add the random-classifier diagonal. Save to outputs/roc_comparison.png.

# Calculate FPR, TPR, and Thresholds
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_probs_knn)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr,roc_auc=auc_lr).plot(ax=ax, name="Logistic Regression")
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn, roc_auc=auc_knn).plot(ax=ax, name="KNN (Scaled)")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curves Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/roc_comparison.png")
plt.close()

print("Saved plot to outputs/roc_comparison.png")
# Add a comment: at the point on each curve where TPR = 0.80, which model has the lower FPR? What does that mean practically — if you needed to catch 80% of positives, which model would produce fewer false alarms?
# COMMENT: At the point where TPR = 0.80, the KNN model has the lower FPR. 
# this means if we need to catch 80% of true positives, the KNN model is  better because it will produce fewer False Positives than Logistic Regression model.

# ==========================================
# --- ROC Question 3 ---
# ==========================================
print("\n-----ROC Q3-----\n")

# Using the logistic regression from Q1, find the threshold that achieves the highest F1 score on the test set. To do this:

# 1. Get fpr, tpr, and thresholds from roc_curve(y_test, y_probs_lr). (did in Q2)
# 2. For each threshold, compute y_pred = (y_probs_lr >= threshold).astype(int) and calculate the F1 score.
# 3. Print the threshold, TPR, FPR, and F1 at the optimum.

f1_scores = []

for threshold in thresholds_lr:
    y_pred = (y_probs_lr >= threshold).astype(int)
    f1_scores.append(f1_score(y_test,y_pred))

threshold_df = pd.DataFrame({
"threshold": thresholds_lr,
"fpr":       fpr_lr,
"tpr":       tpr_lr,
"f1":        f1_scores
})

idx = threshold_df['f1'].idxmax()
 
print(threshold_df)
print("Optimum Threshold Metrics:")
print(threshold_df.iloc[idx].round(3))

# Add a comment: how does this optimal threshold compare to the default 0.5? In a real application, when would you choose a threshold lower than 0.5?
# COMMENT: The optimal threshold shifts away from the default 0.5 because 0.5 doesn't always give the best balance for the data. 
# In real life, I would choose a threshold lower than 0.5 if missing a positive case (a False Negative) is super risky. 


# ==========================================
# --- GridSearchCV ---
# ==========================================
print("\n-----GridSearchCV Q1-----\n")

# Build a Pipeline with a StandardScaler and a LogisticRegression(max_iter=1000). Use GridSearchCV with cv=5 and scoring="roc_auc" to search over C values [0.001, 0.01, 0.1, 1.0, 10.0, 100.0].

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

y_probs_grid = grid_search.predict_proba(X_test)[:, 1]
new_test_auc = roc_auc_score(y_test, y_probs_grid)

default_test_auc = roc_auc_score(y_test, y_probs_lr)

print(f"Best C value: {grid_search.best_params_['clf__C']}")
print(f"Best CV AUC: {grid_search.best_score_:.3f}")
print(f"Test AUC (Grid Search): {new_test_auc:.3f}")
print(f"Test AUC (Default C=1.0): {default_test_auc:.3f}")

# Add a comment: did the grid search pick the same C you would have guessed by default? By how much did the test AUC change compared to the default C= 1.0?
# COMMENT: The grid search didn't pick the default C=1.0, it went with C=100.0 instead. The test AUC didn't change at all—it stayed exactly at 0.706. 
# So even though it found a "better" C during cross-validation, it made zero difference on the actual test data.

# ==========================================
# --- GridSearchCV Question 2 ---
# ==========================================
print("\n-----GridSearchCV Q2-----\n")

# Run a second grid search using the same Pipeline, but this time replace the LogisticRegression with a DecisionTreeClassifier(random_state=42) and search over max_depth values [2, 3, 5, 8, None].
pipe2 = Pipeline([
   ("scaler", StandardScaler()),
   ("clf", DecisionTreeClassifier(random_state=42))
])

param_grid2 = {
    "clf__max_depth": [2, 3, 5, 8, None]
}
grid_search2 = GridSearchCV(
    estimator=pipe2,
    param_grid=param_grid2,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
)
grid_search2.fit(X_train, y_train)

#predict
y_probs_tree = grid_search2.predict_proba(X_test)[:, 1]
test_auc_tree = roc_auc_score(y_test, y_probs_tree)

# Print the best max_depth and best CV AUC, then print the test AUC.
print(f"Max Best Depth: {grid_search2.best_params_['clf__max_depth']}")
print(f"Best CV AUC: {grid_search2.best_score_:.3f}")
print(f"Test AUC (Decision Tree): {test_auc_tree:.3f}")

# Add a comment: compare the best AUC from Q1 (logistic regression) to this one (decision tree). Which model would you bring into further development? Is AUC the only thing you would consider?
# COMMENT: The Decision Tree beat the Logistic Regression (0.935 vs 0.706), so I'd  bring the Decision Tree into further development. But AUC isn't the only thing that matters. 
# I also have to think about how easy the model is to explain to people and make sure it isn't just memorizing the data.

# ==========================================
# --- GridSearchCV Question 3 ---
# ==========================================
print("\n-----GridSearchCV Q3-----\n")

# Look at the cv_results_ from either grid search. Print the mean and standard deviation of the CV AUC for each parameter value, sorted from best to worst.

results = grid_search2.cv_results_
means = results['mean_test_score']
stds = results['std_test_score']
params = results['params']

# Zip and sort 
combined = list(zip(means, stds, params))
combined.sort(key=lambda x: x[0], reverse=True)

print("Decision Tree CV Results (Best to Worst):")
for mean, std, param in combined:
    print(f"Mean AUC: {mean:.3f} | Std Dev: {std:.3f} | Params: {param}")

# COMMENT: If two settings have the same mean score, I'd go with the one that has the lower standard deviation. 
# A lower standard deviation means the model's performance is more stable and consistent no matter how the data is split up. 


# ==========================================
# --- joblib ---
# ==========================================
print("\n-----joblib  Q1-----\n")

# Take the best Pipeline from GridSearch Question 1 (the logistic regression pipeline). 
# Save it to models/warmup_model.pkl using joblib.dump. Then load it back in the same script using joblib.load and confirm it makes identical predictions to the original:

#save
best_lr_pipe = grid_search.best_estimator_
joblib.dump(best_lr_pipe, "models/warmup_model.pkl")
print("Model saved.")

#load
loaded_clf = joblib.load("models/warmup_model.pkl")
original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# Add a comment: what would break if you saved only the logistic regression model (without the scaler) and then called .predict(X_test) on the loaded model, where X_test is unscaled?
# COMMENT: If you only saved the logistic regression model and forgot the scaler, everything would break. The model was trained on scaled data, so it expects the numbers to be small. 
# If you try to feed it raw, unscaled test data later, it will not match and fail.

# ==========================================
# --- joblib Question 2 ---
# ==========================================
print("\n-----joblib  Q2-----\n")

# Demonstrate a minimal version of the train/predict split. In the same warmup_04.py:
# Save the best logistic regression Pipeline to models/warmup_model.pkl (you may have already done this in Q1).
# Add a clearly labeled section — use a comment like # --- Simulated prediction script ---.
# In that section, load the model fresh from disk and use it to predict on three manually constructed rows:

# --- Simulated prediction script ---

#load
simulated_model = joblib.load("models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

new_preds = simulated_model.predict(new_samples)
new_probs = simulated_model.predict_proba(new_samples)

# Print the predicted class and the probability for each row. Add a comment: what do you expect the all-zeros row to predict? Why?
for i in range(len(new_samples)):
    print(f"Row {i+1} Predicted Class: {new_preds[i]} | Probabilities: {new_probs[i]}")

# COMMENT: If we didn't have a scaler in the pipeline, feeding in all zeros would cancel out all the model's feature weights. 
# But since our pipeline scales the data first, those zeros  get shifted based on the training data's averages before hitting the model. 

