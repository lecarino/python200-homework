'''
Part 1: Warmup Exercises
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ==========================================
# --- Preprocessing ---
# ==========================================
print("\n-----Preprocessing Q1-----\n")

# Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42. Print the shapes of all four arrays.
#SPLIT:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f'X-TRAIN SHAPE: {X_train.shape}')
print(f'X-TEST SHAPE: {X_test.shape}')
print(f'Y-TRAIN SHAPE: {y_train.shape}')
print(f'Y-TEST SHAPE: {y_test.shape}')

# ==========================================
print("\n-----Preprocessing Q2-----\n")

# Fit a StandardScaler on X_train and use it to transform both X_train and X_test. Print the mean of each column in X_train_scaled -- they should all be very close to 0. 
# Add a comment explaining in one sentence why you fit the scaler on X_train only.

# SCALER: 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("X_train_scaled column means:")
print(X_train_scaled.mean(axis=0))


# ==========================================
# --- KNN ---
# ==========================================
print("\n-----KNN Q1-----\n")

# Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train), and predict on the test set.
# Print the accuracy score and the full classification report.

#Create
knn = KNeighborsClassifier(n_neighbors=5)
#Fit
knn.fit(X_train,y_train)
#predict
preds= knn.predict(X_test)
#Results:
print(f"KNN Accuracy Score: {accuracy_score(y_test,preds)}")
print(f"KNN Classification Report:\n{classification_report(y_test,preds)}")


# ==========================================
print("\n-----KNN Q2-----\n")
# Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). 
# Print the accuracy score. Add a comment: does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?

#CREATE:
knn_scaled = KNeighborsClassifier(n_neighbors=5)
#FIT:
knn_scaled.fit(X_train_scaled,y_train)
#PREDICT:
preds_scaled = knn_scaled.predict(X_test_scaled)
#RESULTS:
print(f"KNN Scaled Accuracy Score: {accuracy_score(y_test,preds_scaled)}")
print(f"KNN scaledClassification Report:\n{classification_report(y_test,preds_scaled)}")

#COMMENT:
# Scaling actually hurt the performance (dropping accuracy from 1.0 to 0.93). 
# This happens because the unscaled petal features naturally have a wider variance and are highly predictive, so they rightly dominate the KNN distance calculation. 
# Scaling forces the less helpful, "noisy" sepal features to have equal weight, which confuses the model.

# ==========================================
print("\n-----KNN Q3-----\n")

# Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. Print each fold score, the mean, and the standard deviation. 
# Add a comment: is this result more or less trustworthy than a single train/test split, and why?

cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print(cv_scores)
print(f"Mean: {cv_scores.mean():.3f}")
print(f"STD: {cv_scores.std():.3f}")

#COMMENT: Because every training example participates in evaluation at some point, so the average score is more stable.
# The std across all folds is very low and is consistent, so it's not a fluke. 


# ==========================================
print("\n-----KNN Q4-----\n")

# Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each, compute 5-fold cross-validation accuracy on the unscaled training data and print k and the mean CV score. 
# Add a comment identifying which k you would choose and why.

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train,y_train,cv=5)

    print(f"K Value: {k}")
    print(f"MEAN CV Score: {cv_scores.mean():.3f}\n")
print(f"Finished K values loop")

#Comment: I would choose k = 5 or 7 since they give out the highest mean cv score. I would choose k=7 because it has more neighbors and would generalize data better. 


# ==========================================
# --- Classifier Evaluation ---
# ==========================================
print("\n-----CE Q1-----\n")

# Using your predictions from KNN Question 1, create a confusion matrix and display it with ConfusionMatrixDisplay, passing display_labels=iris.target_names. 
# Save the figure to outputs/knn_confusion_matrix.png. Add a comment: which pair of species does the model most often confuse (if any)?

cm = confusion_matrix(y_test,preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= iris.target_names)

disp.plot()
plt.title("KNN Confusion Matrix Iris")
plt.savefig("outputs/knn_confusion_matrix.png")
print("\n saved plot to outputs/knn_confusion_matrix.png \n")

#COMMENT: The model doesnt get confused with any species as it is labeled 0 for each off-diagonal number.


# ==========================================
# --- The sklearn API: Decision Trees ---
# ==========================================
print("\n-----DT Q1-----\n")

# Create a DecisionTreeClassifier(max_depth=3, random_state=42), fit it on the unscaled training data, and predict on the test set. 
    #create
dt = DecisionTreeClassifier(max_depth=3,random_state=42)
    #fit
dt.fit(X_train,y_train)
    #predict
dt_preds = dt.predict(X_test)

# Print the accuracy score and classification report. 
    #print
print(f"Decision Tree Accuracy Score: {accuracy_score(y_test,dt_preds)}")
print(f"Decision Tree Classification Report:\n{classification_report(y_test,dt_preds)}")

# Add a comment comparing the Decision Tree accuracy to KNN. 
    #COMMENT: The decision tree accuracy score is 0.967, while the KNN_scaled accuracy score is 0.933, and the KNN unscaled is 1.00. The decision tree is slightly more accurate than the KNN_scaled but lower than KNN.
# Then add a second comment: given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
    #COMMENT:  I think scaling the data for decision tree would not matter since scaling is for distance calculations and numbers, while decision trees dont rely on them. 


# ==========================================
# --- Logistic Regression and Regularization ---
# ==========================================
print("\n-----LR & R Q1-----\n")

# Train three logistic regression models on the scaled Iris data, identical in every way except for the C parameter: C=0.01, C=1.0, and C=100. Use max_iter=1000 and solver='liblinear' for all three. 
# For each model, print the C value and the total size of all coefficients using np.abs(model.coef_).sum(). 
# Add a comment: what happens to the total coefficient magnitude as C increases? What does this tell you about what regularization is doing?

# Create
log_reg_1 = OneVsRestClassifier(LogisticRegression(C=0.01, max_iter=1000, solver='liblinear'))
# Fit
log_reg_1.fit(X_train_scaled, y_train)
coef_sum_1 = sum(np.abs(est.coef_).sum() for est in log_reg_1.estimators_)
# Print
print(f"C Value (0.01): {coef_sum_1:.3f}")

#Create
log_reg_2 = OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
#fit
log_reg_2.fit(X_train_scaled,y_train)
coef_sum_2 = sum(np.abs(est.coef_).sum() for est in log_reg_2.estimators_)
print(f"C Value (1.0): {coef_sum_2:.3f}")

#Create
log_reg_3 = OneVsRestClassifier(LogisticRegression(C=100, max_iter=1000, solver='liblinear'))
#fit
log_reg_3.fit(X_train_scaled, y_train)
coef_sum_3 = sum(np.abs(est.coef_).sum() for est in log_reg_3.estimators_)
print(f"C Value (100): {coef_sum_3:.3f}")
#Comment: As C increases, the total coefficient magnitude increases as well. C value is inverse to regularization strength. A small C applies strong regularization, shrinking the coefficients toward zero to prevent overfitting. A large C applies weak regularization, allowing the coefficients to grow larger to fit the training data.

# ==========================================
# --- PCA ---
# ==========================================
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


print("\n-----PCA Q1-----\n")
# Print the shape of X_digits and images. Then create a 1-row subplot showing one example of each digit class (0-9), using cmap='gray_r' with each digit's label as the title. 
# Save the figure to outputs/sample_digits.png. 
# (gray_r is the reversed grayscale colormap -- it renders higher pixel values as darker, so digits appear as dark ink on a light background, which is more readable than the default.)

print(f"X_digits shape: {X_digits.shape}")
print(f"Images shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0] 
    
    axes[digit].imshow(images[idx], cmap="gray_r")
    axes[digit].set_title(f"Label: {digit}")
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
print("\nSaved plot to outputs/sample_digits.png\n")

# ==========================================
print("\n-----PCA Q2-----\n")

# Fit PCA() on X_digits (with no n_components argument) then get the scores with scores = pca.transform(X_digits). 
#Create
pca = PCA()
#FIT
pca.fit(X_digits)
#Scores
scores = pca.transform(X_digits)
# As in the lesson, scores tell you how strongly each component is weighted for each sample -- scores[i, 0] is the weighting for PC1 in sample i, scores[i, 1] is the weighting for PC2, and so on.

# Use scores[:, 0] and scores[:, 1] to make a scatter plot, coloring each point by its digit label and adding a colorbar. Here is the pattern for coloring by a label array and attaching a colorbar:
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.title("PCA 2D Projection of Digits")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")


# Save the figure to outputs/pca_2d_projection.png. Add a comment: do same-digit images tend to cluster together in this 2D space?
plt.savefig("outputs/pca_2d_projection.png")
print("\nSaved plot to outputs/pca_2d_projection.png\n")
#Comment: Yes, same-digit images do tend to cluster together

# ==========================================
print("\n-----PCA Q3-----\n")

# Using the PCA object you fit in Question 2, plot cumulative explained variance vs. number of components using np.cumsum(pca.explained_variance_ratio_). 
# Save to outputs/pca_variance_explained.png. Add a comment: approximately how many components do you need to explain 80% of the variance?

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("outputs/pca_variance_explained.png")
plt.title("PCA Explained Variance")
plt.close()
print("\nSaved plot to outputs/pca_variance_explained.png\n")

#COMMENT: About 13-14 components to explain 80% of the variance


# ==========================================
print("\n-----PCA Q4-----\n")

# The preprocessing lesson showed that a reconstruction is built by starting from the mean and adding each component weighted by its score.
# Here is the same idea generalized to n components -- add this function to your file:
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Using this function, the PCA object, and the scores from Question 2, reconstruct the first 5 digits in X_digits using reconstruction through principal components n = 2, 5, 15, and 40.
n_values = [2, 5, 15, 40]

#5x5 grid (1 row for Originals, 4 rows for the different 'n' values)
fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 10))

for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    axes[0, col].set_title(f"Original")
    axes[0, col].axis("off")

# ROWS 1-4: Plot the reconstructions
for row, n in enumerate(n_values, start=1):
    for col in range(5):
        reconstructed_image = reconstruct_digit(col, scores, pca, n)
    
        #plot
        axes[row, col].imshow(reconstructed_image, cmap="gray_r")
        axes[row, col].set_title(f"n={n}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()
print("Saved plot to outputs/pca_reconstructions.png")

# COMMENT: At n=2 and n=5, the images are mostly blurry blobs. By n=15, the digits become clearly recognizable as numbers. This makes sense because, as seen in Q3, 14-15 components capture about 80% of the variance in the dataset. By n=40, the reconstructions look almost exactly like the original images.

