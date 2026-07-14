'''
Part 2: Mini-Project -- Predicting Student Math Performance
'''

# ==========================================
# --- Pre-preprocessing ---
# ==========================================

# Notice how fields are separated. Notice which values are quoted and which are not. Look at what G1, G2, and G3 look like in the raw file. If you were loading this with pd.read_csv(),
# what parameter would you need to specify beyond the filename? Write that observation as a comment at the top of your script before you write the load call.
#COMMENT: The parameter i would need to specify beyond the filename would be the semicolon (;). pd.read_csv() by default looks for commas. 


# ==========================================
# --- IMPORTS ---
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# ==========================================
# --- Task 1: Load and Explore ---
# ==========================================

# Load the dataset with the correct separator. Print the shape, the first five rows, and the data types of all columns.
# Load: 
print("\n-----TASK 1:-----\n")


df = pd.read_csv("student_performance_math.csv", sep=";")

#Print:
print(f"Shape: {df.shape}\n")
print(f"First 5 rows:\n{df.head()}\n")
print(f"Data Types:\n{df.dtypes}\n")

# Then plot a histogram of G3 with 21 bins (one per possible value, 0-20). Add a title "Distribution of Final Math Grades", label both axes, and save to outputs/g3_distribution.png
# Plot:
plt.figure()
plt.hist(x= df["G3"], bins= 21, edgecolor='black')
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("outputs/g3_distribution.png")
plt.close()

print("Plot saved to outputs/g3_distribution.png")

# ==========================================
# --- Task 2: Preprocess the Data ---
# ==========================================
print("\n-----TASK 2:-----\n")

# Handle the G3=0 rows first. Filter them out and save the result to a new DataFrame. 
# Print the shape before and after to confirm how many rows were removed. Add a comment explaining your reasoning -- why would keeping these rows distort the model?
print(f"Before: {df.shape}\n")

new_df = df[df['G3'] != 0].copy()
print(f"After: {new_df.shape}\n")

#Comment: Keeping the rows would distort the model because the students who have a zero means they didnt take the final exam and would affect the scores of the data where the students actually went and took it.

# Then convert the yes/no columns to 1/0 and the sex column to 0/1.
#sex:
new_df['sex'] = new_df['sex'].replace({'M':1, 'F':0})

#yes/no cols:
yn_cols = ['schoolsup', 'internet', 'higher', 'activities']
new_df[yn_cols]= new_df[yn_cols].replace({'yes':1, 'no':0})

print(new_df)

# Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one, and print both values. The difference is striking.
og_df_corr = df['absences'].corr(df['G3'])
print(f"Original Correlation: {og_df_corr:.4f}")

new_df_corr = new_df['absences'].corr(new_df['G3'])
print(f"Filtered Correlation: {new_df_corr:.4f}")

# Add a comment explaining why filtering changes the result: what were students with G3=0 doing in the original data that made absences look like a weak predictor?
# COMMENT: The G3=0 in the original df makes it seem like if you miss class, it won't affect your grade. but taking it out makes a huge a difference that shows absences being a strong predictor.


# ==========================================
# --- Task 3: Exploratory Data Analysis ---
# ==========================================
print("\n-----TASK 3:-----\n")

# Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset, and print them sorted from most negative to most positive. 
# Which feature has the strongest relationship with G3? Are any results surprising?

g3_correlations = new_df.corr(numeric_only = True)["G3"].sort_values()
print(f"Correlations with G3:\n{g3_correlations}\n")

#COMMENT: Failures has the strongest relationship with G3. Not surprising at all because if you fail a lot then obviously you'll most likely get low grades for G3.

# Then create at least two visualizations of your own choosing and save them to outputs/. Use your judgment from previous weeks of data engineering to guide your use of plots. 
# Use the correlation results to guide you -- what relationships seem worth a closer look? Add a comment for each plot describing what you see.

#VISUALIZATION 1: Failures vs G3
plt.figure()
sns.boxplot(data=new_df, x='failures', y='G3') 

plt.title("Final Grades by Number of Past Failures")
plt.xlabel("Number of Past Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/failures_boxplot.png")
plt.close()
print('saved plot on outputs/failures_boxplot.png\n')

#COMMENT :Students with 0 past failures have a median grade of around 11.5, which steadily drops as the number of failures increases. 
# By the time a student has 3 past failures, their median grade falls to around 8, showing that past failures are a strong indicator of lower final grades.

#VISUALIZATION 2: Absences vs G3
plt.figure()
sns.scatterplot(data=new_df, x="absences", y="G3", alpha=0.5)

plt.title("Impact of Absences on Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("outputs/absences_scatter.png")
plt.close()
print('saved plot on outputs/absences_scatter.png\n')

#COMMENT: There is a negative trend where students with the fewest absences (0-5) tend to achieve the widest range of grades, including the highest scores (15-20). 
# As absences increase beyond 10-15 days, we see almost no students achieving top-tier grades, demonstrating how high absenteeism limits academic performance.

# ==========================================
# --- Task 4: Baseline Model ---
# ==========================================
print("\n-----TASK 4:-----\n")

# Build the simplest possible model: use failures alone to predict G3. Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print the slope, RMSE, and R² on the test set.

print(new_df['failures'].shape)
#REshape
X = new_df['failures'].values.reshape(-1,1)
#Split
X_train, X_test, y_train, y_test = train_test_split(X, new_df["G3"],test_size=0.2, random_state=42)
#FIt
model = LinearRegression()
model.fit(X_train,y_train)


print(f"Slope: {model.coef_}")

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"RMSE: {rmse:.2f}")
print(f'r2: {r2:.4f} ')

# Add a comment: given that grades are on a 0-20 scale, what do the slopes and RMSE tell you in plain English? Is R² better or worse than you expected from exploratory data analysis?
# Comment: SLope = -1.43, meaning that for every failure, the grade linearly goes down by 1.43. RMSE = 2.96, meaning my model's prediction is off by about 2.96 points on a 20 point scale
# R2 is 0.0895 which is very low, meaning that past failures are part of the story as to why the grades are low, but there are other factors that the model is missing.


# ==========================================
# --- Task 5: Build the Full Model ---
# ==========================================
print("\n-----TASK 5:-----\n")

# Now build a regression model using all of the numeric and binary features from the Feature Guide:
df_clean = new_df.copy()

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, and print both train R² and test R², as well as RMSE on the test set. 
# Compare the test R² to your baseline from Task 4 -- how much does adding more features help?

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
#FIt
model = LinearRegression()
model.fit(X_train,y_train)
#Print:
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"RMSE: {rmse:.2f}")
print(f'r2: {r2:.4f} \n')

#COMMENT: It made the rmse from task 4 go from 2.96 down to 2.86, meaning adding the features helped the model be slightly more accurate. R2 doubled meaning the added features added variance to the model

# Print each feature name alongside its coefficient:
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Look carefully at the coefficients. Sort them mentally from largest to smallest. Are any signs (positive or negative) surprising given what you know about the data? 
# For any surprising result, add a comment with your best explanation. Then compare train R² to test R² -- are they close, or is there a gap? What does that tell you about the model?

coef_series = pd.Series(model.coef_, index=feature_cols)
sorted_coefs = coef_series.sort_values(ascending=False)

print("\nCoefficients sorted from most positive to most negative:")
print(sorted_coefs)
#COMMENT:Surprisingly, the schoolsup feature is lowest, but I'm guessing there's a bias because those who get school tutoring are usually those who aren't doing well. 

#CoMPARE train R2 and test R2:
train_r2 = model.score(X_train,y_train)
test_r2 = model.score(X_test,y_test)

print(f"\nTrain R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
#COMMENT: train is 0.1749 and test is 0.1539, so there is a small gap between them, meaningn that the model is generalized well. 

# Finally, add a comment answering: if you were deploying this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.
#COMMENT: I would keep internet, higher, failures, and schoolsup because they produce a logical relationship with academic success and demonstrate consistent predictive value.
# I would drop activities, and freetime since they have coefficients close to 0, meanign they have no predictive power in the model.


# ==========================================
# ---Task 6: Evaluate and Summarize ---
# ==========================================
print("\n-----TASK 6:-----\n")
'''A useful way to evaluate a regression model visually is a predicted vs actual plot. 
This is a scatter plot where each point in the test set becomes a dot, with the model's prediction (y_hat) on the x-axis and the true value (y) on the y-axis. 
If the model were perfect, every point would fall exactly on the diagonal (predicted = actual). Clusters or curves away from the diagonal reveal systematic errors that RMSE alone won't show you. 
Random scattering around the diagonal is expected, and acceptable, prediction error.'''

# Create this plot for your test set. Add a diagonal reference line (for y=y_predicted), a title "Predicted vs Actual (Full Model)", labeled axes, and save to outputs/predicted_vs_actual.png.
plt.figure()
plt.scatter(y_pred,y_test,label='Student Scores')
plt.plot([0,20],[0,20],color='red',label='Perfect Prediction')
plt.title('Predicted vs Actual (Full Model)')
plt.xlabel("Predicted Grade (y_hat)")
plt.ylabel("Actual Grade (y)")
plt.legend()

plt.savefig("outputs/predicted_vs_actual.png")
plt.close()
print('Saved plot to outputs/predicted_vs_actual.png\n')

# Add a comment: does the model seem to struggle more at the high end, the low end, or is error roughly uniform across grade levels? What does a value above or below the diagonal mean?
# COMMENT: The error is roughly uniform across grade levels. The value above means the actual score is higher than the perfect prediction, while the values below means that they're lower than the perfect prediction.

# Then write a plain-language summary in your comments statements covering:

# The size of the filtered dataset and the test set
# The RMSE and R² of your best model in plain language -- on a 0-20 scale, what does a typical prediction error actually mean?
# Which two features have the largest positive and largest negative coefficients, and what those mean
# One result that surprised you

#COMMENTS: 
# The filtered dataset contains 357 students, and I reserved 20% of those (71 students) for the test set.
# My RMSE is 2.86, meaning my model's predictions are typically off by almost 3 points on a 20-point scale. 
# My R2 is 0.15, meaning this model only explains about 15% of why student grades vary, so it is missing a lot of outside factors.
# The largest positive feature is the internet feature meaningn those who have access to internet have a better chance of getting higher grades. 
# The largest negative coefficient is those with schoosup, meaning those who are already doing bad already have a bad grade.
# One result that surprised me was the 'schoolsup' feature but I did realize that those who get supplementary help are usually those who struggle already.


# Neglected Feature: The Power of G1
# Add G1 (first period grade) as a feature to the full model from Task 5 and refit. 
# We kept it out because it is so powerful. Print the new test R². The jump will be large -- from roughly 0.30 to somewhere around 0.80.

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                "internet", "sex", "freetime", "activities", "traveltime", "G1"] #Added G1 in features column!
X = df_clean[feature_cols].values
y = df_clean["G3"].values

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
#FIt
model = LinearRegression()
model.fit(X_train,y_train)
#Print:
y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"RMSE: {rmse:.2f}")
print(f'r2: {r2:.4f} \n')

# Add a comment addressing these questions: does a high R² here mean G1 is causing G3? 
# Is this a useful model for identifying students who might struggle? What might educators need to do if they wanted to intervene early, before G1 is even available?
# COMMENT: I don't think a high R2 here for G1 is causing G3. I think those who do well in G1 do well in G3 because of their study habits and work ethic which causes them to carry it over to G3. It is a useful model and also not. a student could have a bad grade in G1 and try to study harder or get help to get better at G3. And vice versa, maybe a student who does well in G1 might get lazy and complacent for G3. If they want to intervene before G1 is available, then they might have to use my model with all the features before getting the grade of G1, since it may help predict who does well and who doesn't. 