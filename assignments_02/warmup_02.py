''' Part 1: Warmup Exercises'''



# ==========================================
# --- The scikit-learn API ---
# ==========================================

# scikit-learn Question 1: The core pattern in scikit-learn is create → fit → predict. Practice it here with a simple dataset: years of work experience versus annual salary.
print("\n-----SL Q1-----\n")
import numpy as np
from sklearn.linear_model import LinearRegression

years = np.array([1,2,3,5,7,10]).reshape(-1,1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
# Create a LinearRegression model, fit it to this data, and then predict the salary for someone with 4 years of experience and someone with 8 years. 
# Print the slope (model.coef_[0]), the intercept (model.intercept_), and the two predictions. Label each printed value.
new_x = np.array([4,8]).reshape(-1,1)

model = LinearRegression()
model.fit(years,salary)
y_predicted = model.predict(new_x)

print(f"The slope = {model.coef_[0]} \nThe intercept = {model.intercept_} \nThe preditions for 4 years = ${y_predicted[0]:.2f} and 8 years = ${y_predicted[1]:.2f} ")

# ---------------------------------------

# scikit-learn Question 2: scikit-learn requires the feature array X to be 2D even when you only have one feature. Start with this 1D array:
print("\n-----SL Q2-----\n")
x = np.array([10, 20, 30, 40, 50])

# Print its shape. Use .reshape() to convert it to a 2D array and print the new shape. Add a comment explaining, in your own words, why scikit-learn needs X to be 2D.
print(x.shape)

new_x = x.reshape(-1,1)
print(new_x.shape)

#COMMENT: It needs to be 2D to take into consideration the (num_samples,num_features) even if you only have 1 feature. A 1D array would have caused an error. 

# ---------------------------------------

# scikit-learn Question 3: K-Means is an unsupervised algorithm that follows the same create → fit → predict pattern as everything else in scikit-learn. 
# Use the code below to generate a synthetic dataset with three natural clusters:
print("\n-----SL Q3-----\n")
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)


# Create a KMeans model with n_clusters=3 and random_state=42, fit it to X_clusters, and predict a cluster label for each point. 
# Print the cluster centers (kmeans.cluster_centers_) and how many points fell into each cluster using np.bincount(labels).

# 1. CREATE:
kmeans = KMeans(n_clusters=3, random_state=42)
# 2. FIT:
kmeans.fit(X_clusters)
# 3. PREDICT:
labels = kmeans.predict(X_clusters)

print(f"Cluster centers:\n {kmeans.cluster_centers_}\nPoints in each cluster: {np.bincount(labels)}")

# Then create a scatter plot coloring each point by its cluster label, plot the cluster centers as black X's, add a title and axis labels. 
# Save the figure to outputs/kmeans_clusters.png.

plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='black', s=100, linewidth=3)
plt.title('K-Means Clustering')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

# ==========================================
# --- Linear Regression ---
# ==========================================

# The questions below all use the same synthetic medical costs dataset: 
# 100 patients, each with an age (20 to 65), a smoker flag (0 = non-smoker, 1 = smoker), and an annual medical cost as the target. Generate it once and reuse the variables throughout

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age = np.random.randint(20,65,num_patients).astype(float)
smoker = np.random.randint(0,2,num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0,3000,num_patients)

# ---------------------------------------
# Linear Regression Question 1
print("\n-----LR Q1-----\n")

# Before fitting anything, look at the data. Create a scatter plot of age on the x-axis and cost on the y-axis. 
# Color the points by smoker status by passing c=smoker and cmap="coolwarm" to plt.scatter(). 
# Add a title "Medical Cost vs Age", label both axes, and save to outputs/cost_vs_age.png.

plt.figure()
plt.scatter(x=age,y=cost,c=smoker,cmap='coolwarm')
plt.title("Medical Cost vs Age")
plt.xlabel('Age')
plt.ylabel('Cost')
plt.savefig("outputs/cost_vs_age.png")
plt.close()
print("SAVED cost_vs_age.png FIGURE TO OUTPUTS")

# Add a comment describing what you see. Are there two distinct groups visible? What does that suggest about the smoker variable?
#COMMENT: There are two distinct groups visible. The smoker variable is seen to affect the medical cost by being higher in cost regardless of age. 

# ---------------------------------------
# Linear Regression Question 2
print("\n-----LR Q2-----\n")

# Split the data into training and test sets using age as the only feature, an 80/20 split, and random_state=42. Reshape age to a 2D array before using it as X. Print the shapes of all four arrays.
X = age.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,cost,test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# ---------------------------------------
# Linear Regression Question 3
print("\n-----LR Q3-----\n")

# Fit a LinearRegression model to your training data from Question 2. Print the slope and intercept. Then predict on the test set and print:
# RMSE: np.sqrt(np.mean((y_pred - y_test) ** 2))
# R² on the test set: model.score(X_test, y_test)
# Add a comment interpreting the slope in plain English -- what does it mean for medical costs?

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(f"Slope:, {model.coef_[0]:.2f}")
print(f"Intercept:, {model.intercept_:.2f}")

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"RMSE:, {rmse:.2f}")
print(f'r2:, {r2:.4f} ')

#COMMENT: The slope being 196.58 means that for every time the age of a subject increases, the avg cost of medical bills infcrease by $196.58. 

# ---------------------------------------
# Linear Regression Question 4
print("\n-----LR Q4-----\n")

# Now add smoker as a second feature and fit a new model.
X_full = np.column_stack([age,smoker])
print(f'X_full shape: {X_full.shape}')
# Split, fit, and print the test R². Compare it to the R² from Question 3 -- does adding the smoker flag help? Print both coefficients:
# SPLIT:
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

# FIT: 
model_full = LinearRegression()
model_full.fit(X_full_train,y_full_train)

#R2:
r2_full= model_full.score(X_full_test,y_full_test)
print(f'r2:, {r2_full:.4f} ')

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Add a comment interpreting the smoker coefficient: what does it represent in practical terms?
# COMMENT: Smoker coefficient brings up the cost to a whopping $14538.04!  Since it's a binary number, being a smoker vs not being one increases the difference by about $14538. 

# ---------------------------------------
# Linear Regression Question 5
print("\n-----LR Q5-----\n")

# A predicted vs actual plot is a standard tool for evaluating regression models. Each test observation becomes a dot: the model's prediction goes on the x-axis, the true value goes on the y-axis. 
# A perfect model would place every point on the diagonal line where predicted equals actual.

# Using the two-feature model from Linear Regression Question 4, create this plot for the test set. 
# Add a diagonal reference line, a title "Predicted vs Actual", labeled axes, and save to outputs/predicted_vs_actual.png.

#PREEDICT:
y_full_pred = model_full.predict(X_full_test)

#PLOT: 
min_val = y_full_test.min()
max_val = y_full_test.max()

plt.figure()
plt.scatter(y_full_pred,y_full_test)
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
plt.title("Predicted vs Actual")
plt.xlabel("Model's Prediction")
plt.ylabel("True Values")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

print("Plot has been saved to outputs/predicted_vs_actual.png")

#  what does it mean when a point falls above the diagonal? What about below?
#COMMENT: If a point falls above, then the model underpredicted and if it was below, then the model overpredicted. 