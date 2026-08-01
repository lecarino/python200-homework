'''
Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout
'''
print("\n-----Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout-----\n")


# ==========================================
# --- Task 1: Load and Explore ---
# ==========================================
print("\n-----Task 1-----\n")
# The logistic regression lesson shows exactly how to load this dataset. Adapt that code for your script.

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)
# Pasted Column names
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

#Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

#print head()
df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES

# Once it is loaded, take some time to understand what you are working with. 
# How many emails are in the dataset? How balanced are the two classes? What does that balance (or imbalance) mean for how you should interpret a raw accuracy score?
print(f"Total emails: {len(df)}")
print("\nClass Balance:")
print(df['spam_label'].value_counts())

# Comment: [4601 rows x 58 columns]. spam_label == 1: [1813 rows x 58 columns]. spam_label == 0 [2788 rows x 58 columns]. There are more non spam mails. 
# raw accuracy can be misleading in imbalanced datasets, so we have to look at other metrics (like Precision and Recall) to know if a model is actually smart.

# Now explore how a few key features differ between spam and ham. 
# For each of word_freq_free, char_freq_!, and capital_run_length_total, create a boxplot showing the distribution of that feature for spam emails versus ham emails.
# Save them to outputs/. What do you notice? Are the differences between classes dramatic or subtle?

features = ['word_freq_free','char_freq_!','capital_run_length_total' ]


for feature in features:
    spam = df[feature][df["spam_label"]==1]
    ham =  df[feature][df["spam_label"]!=1]

    plt.figure()
    plt.boxplot([ham, spam],tick_labels=['Ham','Spam'])
    plt.title(f"Distribution of {feature}")
    plt.savefig(f'outputs/{feature}_boxplot.png')
    plt.close()
    print(f"\nSaved outputs/{feature}_boxplot.png\n")

# spam has much higher extreme values for these features than ham.

# Then look at the raw scale of the features more broadly. 
# Notice that many emails have a value of zero for most word-frequency features -- most emails do not contain the word "free" at all. 
# What does this heavy skew toward zero tell you about the data? 
# Why does the numeric scale vary so dramatically across features (some are tiny fractions, others reach into the thousands)? Why might that matter for some of the models you are about to build?

#COMMENT: the vast majority of columns for any given row will simply be zero because of sparsity. The numeric scale varies because not all of the features measure the same thing with the same units.
# It matters to know because some algorithms rely on mathemtics and units while others are blind to them.


# ==========================================
# ---Task 2: Prepare Your Data ---
# ==========================================
print("\n-----Task 2-----\n")
# Before building any models, prepare your data for the experiments in Task 3. 
# You will need a train/test split and will need to think about how to handle the feature scales you noticed in Task 1. Document your choices in comments.

#Clean X and y
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

#SPLIT:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f'X-TRAIN SHAPE: {X_train.shape}')
print(f'X-TEST SHAPE: {X_test.shape}')
print(f'Y-TRAIN SHAPE: {y_train.shape}')
print(f'Y-TEST SHAPE: {y_test.shape}')

#SCALER: 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#PCA PREPROCESSING
# Fit PCA on the training data only -- same reason as the scaler: fitting on all the data lets test-set information leak into the components.

pca = PCA()
pca.fit(X_train_scaled)

# Plot the cumulative explained variance, save it to outputs/, and print n -- the number of components where it first reaches 90%.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

#Plot
plt.figure()
plt.plot(cumulative_variance)
plt.axhline(y=0.90, color='r', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance (Spambase)")
plt.savefig("outputs/pca_spam_variance.png")
plt.close()

n = np.where(cumulative_variance >= 0.90)[0][0] + 1
print(f"Number of components for 90% variance (n): {n}")
# With n determined, transform both sets and slice to the first n components:
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]

print(f'\nX_train_scaled shape: {X_train_scaled.shape}\n')
print(f'X_test_scaled{X_test_scaled.shape}\n')
print(f'X_train_pca{X_train_pca.shape}\n')
print(f'X_test_pca{X_test_pca.shape}\n')

# ==========================================
# ---Task 3: A Classifier Comparison---
# ==========================================
print("\n-----Task 3-----\n")

# Build and evaluate the following five classifiers. For each, print the accuracy and the full classification report.

# ==========================================
print("\n----- KNN UNSCALED -----\n")
# KNeighborsClassifier(n_neighbors=5) trained on the unscaled data
# CREATE
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
# FIT
knn_unscaled.fit(X_train,y_train)
# PREDICT 
preds_unscaled = knn_unscaled.predict(X_test)
# PRINT
print(f"Unscaled Accuracy Score: {accuracy_score(y_test,preds_unscaled):.2f}")
print(f"KNN Unscaled Classification Report:\n{classification_report(y_test,preds_unscaled)}\n")

# ==========================================
# KNeighborsClassifier(n_neighbors=5) trained on the scaled data, and again on the PCA-reduced data from Task 2 -- compare the two
print("\n-----KNN SCALED-----\n")
#TRAINED
# CREATE
knn_scaled = KNeighborsClassifier(n_neighbors=5)
# FIT
knn_scaled.fit(X_train_scaled,y_train)
# PREDICT 
preds_scaled = knn_scaled.predict(X_test_scaled)
# PRINT
print(f"Scaled Accuracy Score: {accuracy_score(y_test,preds_scaled):.2f}")
print(f"KNN Scaled Classification Report:\n{classification_report(y_test,preds_scaled)}\n")

#PCA
print("\n-----KNN PCA-----\n")
# CREATE
knn_pca = KNeighborsClassifier(n_neighbors=5)
# FIT
knn_pca.fit(X_train_pca,y_train)
# PREDICT 
preds_pca = knn_pca.predict(X_test_pca)
# PRINT
print(f"KNN PCA Accuracy Score: {accuracy_score(y_test,preds_pca):.2f}")
print(f"KNN PCA Classification Report:\n{classification_report(y_test,preds_pca)}\n")

# ==========================================
# DecisionTreeClassifier(random_state=42) -- before settling on a final depth, try max_depth values of 3, 5, 10, and None (unlimited).
#  For each, print both the training accuracy and the test accuracy. What do you notice as depth increases? What does that tell you about overfitting? 
# Pick the depth you would use in production and add a comment explaining your reasoning. 
# Then, using your chosen depth, print the accuracy and full classification report as you did for the other classifiers.

print("\n-----DECISION TREE CLASSIFIER-----\n")
#CREATE
depths = [3, 5, 10, None]
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth,random_state=42)
    #fit
    dt.fit(X_train,y_train)
    #predict
    train_preds = dt.predict(X_train)
    test_preds = dt.predict(X_test)
    #print
    print(f"Decision Tree Training Accuracy Score Depth {depth}: {accuracy_score(y_train,train_preds):.3f}")
    print(f"Decision Tree Test Accuracy Score Depth {depth}: {accuracy_score(y_test,test_preds):.3f}\n")
    

# What do you notice as depth increases? What does that tell you about overfitting? 
# Pick the depth you would use in production and add a comment explaining your reasoning. Then, using your chosen depth, print the accuracy and full classification report as you did for the other classifiers.

# COMMENT: As depth increased from 3 to 10 to None, the training accuracy went all the way up to a perfect 1.0. However, the test accuracy leveled off around 0.91 and stopped improving. This divergence shows the unlimited model was overfitting and perfectly memorizing the training noise. I chose a depth of 10 for production because it gives the highest test accuracy before the model starts severely overfitting.
print("\n-----DECISION TREE CLASSIFIER CHOSEN DEPTH: 10-----\n")
#CHOSEN DEPTH: 10
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
#fit
dt.fit(X_train,y_train)
#predict
preds_dt = dt.predict(X_test)
#print
print(f"DT Depth 10 Accuracy Score: {accuracy_score(y_test,preds_dt):.3f}")
print(f"DT Depth 10 Classification Report:\n{classification_report(y_test,preds_dt)}\n")

# COMMENT (Decision Tree vs KNN & Scaling):
# The Decision Tree (Depth 10) achieved similar accuracy to the Scaled KNN model. 
# However, unlike KNN, Decision Trees do not rely on distance calculations. They simply split data based on feature thresholds. Because of this, scaling the data does not affect a Decision Tree's results at all.



# ==========================================
print("\n-----RANDOM FOREST CLASSIFIER-----\n")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
print(f"RF Depth Accuracy Score: {accuracy_score(y_test,preds_rf):.2f}")
print(f"RF Depth Classification Report:\n{classification_report(y_test,preds_rf)}\n")

dt_importances = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Both the Decision Tree and the Random Forest expose a .feature_importances_ attribute. After building both, 
# print the top 10 most important features for each and save a bar chart of the Random Forest importances to outputs/feature_importances.png. 
# Do the two models agree on which features matter most? Do the results match your intuition about what makes an email spam?

print("Top 10 Decision Tree Features:")
print(dt_importances.head(10))
print("\nTop 10 Random Forest Features:")
print(rf_importances.head(10))

# save a bar chart of the Random Forest importances to outputs/feature_importances.png.
plt.figure()
rf_importances.head(10).plot(kind='bar', color='skyblue')

plt.title("Top 10 Random Forest Feature Importances")
plt.ylabel("Importance")
plt.xticks(rotation=45, ha='right') # Tilts the labels so they are easier to read
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()
print("Saved plot to outputs/feature_importances.png")

# COMMENT: Yes, they have the same top 3 features although different accuracies.  It does match my intuition. 

# ==========================================
print("\n-----LOGISTIC REGRESSION-----\n")

# SCALED
lr_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_scaled.fit(X_train_scaled, y_train)
preds_lr_scaled = lr_scaled.predict(X_test_scaled)

print(f"LR Scaled Accuracy Score: {accuracy_score(y_test, preds_lr_scaled):.3f}")
print(f"LR Scaled Classification Report:\n{classification_report(y_test, preds_lr_scaled)}\n")

# PCA
lr_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
lr_pca.fit(X_train_pca, y_train)
preds_lr_pca = lr_pca.predict(X_test_pca)

print(f"LR PCA Accuracy Score: {accuracy_score(y_test, preds_lr_pca):.3f}")
print(f"LR PCA Classification Report:\n{classification_report(y_test, preds_lr_pca)}\n")

# ==========================================
# --- SUMMARY & BEST MODEL CONFUSION MATRIX ---
# ==========================================
print("\n-----BEST MODEL CONFUSION MATRIX-----\n")

# COMMENT: 
# 1. Best Model: The Random Forest classifier performed the best overall with an accuracy of 0.94.
# 2. PCA vs. Non-PCA: The non-PCA (scaled) models performed just slightly better than their PCA counterparts. However, the PCA models were remarkably close, 
# confirming the hypothesis from Task 2 that PCA successfully captures the most important variance even when the data is highly compressed.
# 3. Optimization Metric: For a spam filter, accuracy is not the best metric. It is much more important to minimize False Positives (legitimate emails marked as spam). 
# If a legit email—like a message from a buyer on Depop or Poshmark—is routed to the spam folder, it could mean a lost sale. 
# A False Negative (a spam email slipping into the inbox) is just a minor annoyance that can be manually deleted.

# Generate the confusion matrix for the best model: (Random Forest)
best_model = rf
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham (0)", "Spam (1)"])

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()
print("Saved plot to outputs/best_model_confusion_matrix.png\n")

# COMMENT: 
# Looking at the confusion matrix (and the classification report), this model makes False Negative errors (predicting Ham when it is actually Spam) more often than False Positive errors. 
# This perfectly aligns with our goal: the model is conservative about sending things to spam, prioritizing the safety of legitimate emails over a perfectly clean inbox.

# ==========================================
# ---Task 4: Cross-Validation---
# ==========================================
print("\n-----Task 4-----\n")

# Using cross_val_score with cv=5, run cross-validation on the training data for each of your classifiers from Task 3.
#  For each, print the mean and standard deviation of the fold scores. 

print("------KNN UNSCALED CV------")
cv_scores_knn_unscaled = cross_val_score(knn_unscaled, X_train, y_train, cv=5)
print(cv_scores_knn_unscaled)
print(f"Mean: {cv_scores_knn_unscaled.mean():.3f}")
print(f"STD:  {cv_scores_knn_unscaled.std():.3f}\n")

print("------KNN SCALED CV------")
# FIX: Use X_train_scaled
cv_scores_knn_scaled = cross_val_score(knn_scaled, X_train_scaled, y_train, cv=5) 
print(cv_scores_knn_scaled)
print(f"Mean: {cv_scores_knn_scaled.mean():.3f}")
print(f"STD:  {cv_scores_knn_scaled.std():.3f}\n")

print("------KNN PCA CV------")
# FIX: Use X_train_pca
cv_scores_knn_pca = cross_val_score(knn_pca, X_train_pca, y_train, cv=5)
print(cv_scores_knn_pca)
print(f"Mean: {cv_scores_knn_pca.mean():.3f}")
print(f"STD:  {cv_scores_knn_pca.std():.3f}\n")

print("------Decision Tree CV------")
cv_scores_dt = cross_val_score(dt, X_train, y_train, cv=5)
print(cv_scores_dt)
print(f"Mean: {cv_scores_dt.mean():.3f}")
print(f"STD:  {cv_scores_dt.std():.3f}\n")

print("------Random Forest CV------")
cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=5)
print(cv_scores_rf)
print(f"Mean: {cv_scores_rf.mean():.3f}")
print(f"STD:  {cv_scores_rf.std():.3f}\n")

print("------Logistic Regression Scaled CV------")
# FIX: Use X_train_scaled
cv_scores_lr_scaled = cross_val_score(lr_scaled, X_train_scaled, y_train, cv=5)
print(cv_scores_lr_scaled)
print(f"Mean: {cv_scores_lr_scaled.mean():.3f}")
print(f"STD:  {cv_scores_lr_scaled.std():.3f}\n")

print("------Logistic Regression PCA CV------")
# FIX: Use X_train_pca
cv_scores_lr_pca = cross_val_score(lr_pca, X_train_pca, y_train, cv=5)
print(cv_scores_lr_pca)
print(f"Mean: {cv_scores_lr_pca.mean():.3f}")
print(f"STD:  {cv_scores_lr_pca.std():.3f}\n")

# Which model is the most accurate? Which is the most stable (lowest variance across folds)? Does the ranking match what you saw with the single train/test split?
# COMMENT ON CROSS-VALIDATION:
# 1. Most Accurate: The Random Forest model is the most accurate because it achieved the highest mean cross-validation score (0.954).
# 2. Most Stable: The Logistic Regression PCA model is the most stable because it has the lowest variance/standard deviation across the 5 folds (0.003).
# 3. Ranking Match: Yes, this ranking matches the single train/test split. Random Forest was the best performer in both, and the Unscaled KNN was the worst in both.

# ==========================================
# ---Task 5: Building a Prediction Pipeline---
# ==========================================
print("\n-----Task 5-----\n")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn5_pipeline = Pipeline([
    ("scaler",     StandardScaler()), # name, object pattern
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn5_pipeline.fit(X_train, y_train)
y_pred = knn5_pipeline.predict(X_test)

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([
    ("scaler",     StandardScaler()),
    ("pca",        PCA(n_components=n)),  # use your n_components from Task 2
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])


# ==========================================
# ---Build your pipelines---
# ==========================================
# Build two pipelines: one for your best tree-based classifier and one for your best non-tree-based classifier. 
# For each, fit on the training data and print the full classification report on the test set. 
# Confirm the results match your earlier manual approach. 
# If your Task 3 experiments showed that PCA improved your non-tree model, include it as a step in that pipeline.


#BEST TREE: Random Forest 
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

#FiT:
rf_pipeline.fit(X_train,y_train)
#predict:
rf_pipeline_predict = rf_pipeline.predict(X_test)
#Print
print(f"Tree Pipeline (Random Forest) Accuracy: {accuracy_score(y_test, rf_pipeline_predict):.3f}")
print(f"Tree Pipeline (Random Forest) Classification Report:\n{classification_report(y_test, rf_pipeline_predict)}\n")


#Best Non Tree: LR Scaled
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
lr_pipeline_preds = lr_pipeline.predict(X_test)

print(f"Non-Tree Pipeline (LR Scaled) Accuracy: {accuracy_score(y_test, lr_pipeline_preds):.3f}")
print(f"Non-Tree Pipeline (LR Scaled) Classification Report:\n{classification_report(y_test, lr_pipeline_preds)}\n")

# Comment on your pipelines: do they have the same structure? Why or why not? What is the practical value of packaging a model this way, especially when handing it off to someone else or deploying it?
# COMMENT:
# 1. Structure: My two pipelines do not have the same structure. 
# 2. Why or why not: The Logistic Regression pipeline includes a StandardScaler step because distance-based and gradient-based models require scaled features to perform correctly. 
#    The Random Forest pipeline only contains the classifier itself because tree-based models split data based on thresholds and are completely unaffected by the numeric scale of the features.
# 3. Practical Value: Packaging a model as a pipeline is incredibly valuable for deployment or sharing. 
# It bundles the preprocessing steps with the model. the next person only needs to pass in the raw data. 
# The pipeline guarantees that the same transformations learned during training are applied to the new data, eliminating the risk of forgetting a step or data leakage.