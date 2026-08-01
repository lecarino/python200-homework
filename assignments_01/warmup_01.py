"""
Python 200 - Week 1 Warmup Exercises
"""

# ==========================================
# Imports
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


# ==========================================
# --- Pandas Review ---
# ==========================================

# Pandas Question 1: Create DataFrame, print first three rows, shape, and data types.
# Print each result with a label (e.g. print(f"Num Rows: {len(df)}")).
print("\n--- Pandas Q1 ---")
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

print(f"First Three Rows: \n{df.head(3)}\n")
print(f"Dataframe Shape: {df.shape}\n")
print(f"Datafrane Data Types:\n {df.dtypes}")

# Pandas Question 2: Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.
passed_students=df[((df["passed"] == True) & (df["grade"] > 80))]
print(f"Students who passed (grade > 80):\n {passed_students}\n")

# Pandas Question 3: Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).
df['grade_curved'] = df["grade"]+ 5
print(f"Updated Curved DF:\n {df}\n")

# Pandas Question 4: Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])
print("\n")

# Pandas Question 5: Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
mean_grades = df.groupby("city")["grade"].mean()
print(f"Mean by city:\n {mean_grades}\n")

# Pandas Question 6: Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df["city"] = df["city"].replace("Austin","Houston")
print(f"New City:\n {df["city"]}\n")

# Pandas Question 7: Sort the DataFrame by "grade" in descending order and print the top 3 rows.
sorted_df = df.sort_values(by="grade",ascending=False)
print(f"Descending Grades:\n {sorted_df.head(3)}\n")


# ==========================================
# --- NumPy Review ---
# ==========================================

# NumPy Question 1: Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
arr1 = np.array([10, 20, 30, 40, 50])
print(f"shape:\n {arr1.shape}")
print(f"dtype:\n {arr1.dtype}")
print(f"ndim:\n {arr1.ndim}")
print("\n")

# NumPy Question 2: Create the following 2D array and print its shape and size (total number of elements).
arr2 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"shape:\n {arr2.shape}")
print(f"size:\n {arr2.size}")
print("\n")

# NumPy Question 3: Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].
arr3 = arr2[:2,:2]
print(arr3)
print("\n")

# NumPy Question 4: Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.
zeros_arr = np.zeros((3,4))
ones_arr = np.ones((2,5))
print(f"Zeros: {zeros_arr},\nOnes: {ones_arr}\n")

# NumPy Question 5: Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
range_arr = np.arange(0, 50, 5)
print(f"Array: {range_arr}")
print(f"Shape: {range_arr.shape}")
print(f"Mean: {range_arr.mean()}")
print(f"Sum: {range_arr.sum()}")
print(f"Std Dev: {range_arr.std():.2f}")

# NumPy Question 6: Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.
normal_arr = np.random.normal(0,1,200)
print(f"Mean: {normal_arr.mean():.4f}, Std Dev: {normal_arr.std():.4f}\n")


# ==========================================
# --- MatPlotLib Review ---
# ==========================================

# Matplotlib Question 1: Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]


plt.figure()
plt.plot(x,y)
plt.title('Squares')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Matplotlib Question 2: Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.figure()
plt.bar(x=subjects,height=scores)
plt.title("Subject Scores")
plt.xlabel('subjects')
plt.ylabel('scores')
plt.show()

# Matplotlib Question 3: Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.title("Scatter Comparison")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Matplotlib Question 4: Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

#Q1
ax1.plot(x, y)
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

#Q2
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

plt.tight_layout()
plt.show()

# ==========================================
# --- Descriptive Review ---
# ==========================================

# Descriptive Stats Question 1: Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
mean = np.mean(data)
median = np.median(data)
var = np.var(data)
std = np.std(data)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Variance: {var:.2f}")
print(f"Standard Deviation: {std:.2f}")

# Descriptive Stats Question 2: Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
norm = np.random.normal(65,10,500)

plt.figure()
plt.hist(norm,20,edgecolor='black')
plt.title("Distribution of Scores")
plt.xlabel('score')
plt.ylabel('frequency')
plt.show()

# Descriptive Stats Question 3: Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()
plt.boxplot(x= [group_a, group_b], tick_labels= ["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

# Descriptive Stats Question 4: You are given two datasets: one normally distributed and one 'exponential' distribution.
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".
# Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.

plt.figure()
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.show()

#COMMENT: The Exponential distribution is more skewed to the right. Mean would be better for central tendency measure for the Normal distribution, while Median is better for the Exponential distribution because it isn't affected by outliers.

# Descriptive Stats Question 5 Print the mean, median, and mode of the following:

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Data1 Mean: {np.mean(data1)}\n")
print(f"Data2 Mean: {np.mean(data2)}\n")

print(f"Data1 Median: {np.median(data1)}\n")
print(f"Data2 Median: {np.median(data2)}\n")

print(f"Data1 Mode: {pd.Series(data1).mode()[0]}\n")
print(f"Data2 Mode: {pd.Series(data2).mode()[0]}\n")

# Why are the median and mean so different for data 2? Add your answer as a comment in the code.
# COMMENT: The median and mean are so different bbecause oif the outlier in data 2 obscuring the mean for data 2. However, since Median resists outliers, it is able to have the same median as data1. 

# ==========================================
# --- Hypothesis Testing Review ---
# ==========================================

# Hypothesis Question 1: Run an INDEPENDENT samples t-test on the two groups below. Print the t-statistic and p-value.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print("Independent T-Test")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")
print('\n')

# Hypothesis Question 2: Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
if p_val > 0.05:
    print("Statistically Insignificant")
else:
    print("The result is statistically significant (reject the null hypothesis).")
print("\n")

# Hypothesis Question 3: Run a PAIRED t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

#paired t-test:
t_stat, p_val = stats.ttest_rel(before,after)
print("Paired T-Test")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}") 
print("\n")

# Hypothesis Question 4: Run a ONE-SAMPLE t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat, p_val = stats.ttest_1samp(scores,70)
print("One Sample T-Test")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4e}")  
print("\n")

# Hypothesis Question 5: Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.
t_stat_1tail, p_val_1tail = stats.ttest_ind(group_a, group_b, alternative='less')
print(f"P-value (one-tailed): {p_val_1tail}")

# Hypothesis Question 6: Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("Group A scored significantly lower than Group B. The P-value is so small that it is not likely this is due to chance")

# ==========================================
# --- Correlation Review ---
# ==========================================

# Correlation Question 1: Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(f"Full Matrix:\n{corr_matrix}\n")
print(f"Correlation Coefficient:\n {corr_matrix[0, 1]}\n")

# What do you expect the correlation to be, and why? Add your answer as a comment in the code.
# COMMENT: I expect the correlation coefficient to be exactly 1 because y value is equal to 2x (y=2x). It is linear

# Correlation Question 2: Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r,p = pearsonr(x,y)
print(f"Correlation Coefficient: {r}")
print(f"P-value: {p}")
print("\n")

# Correlation Question 3: Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df_q3 = pd.DataFrame(people)
print("people df correlation:")
print(df_q3.corr())
print("\n")

# Correlation Question 4: Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()
plt.scatter(x,y)
plt.title("Negative Correlation")
plt.xlabel("x values")
plt.ylabel('y values')
plt.show()

# Correlation Question 5: Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".
plt.figure()
sns.heatmap(df_q3.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# --- Pipelines ---
# ==========================================

# Pipeline Question 1: Implement the following three functions and then connect them in a data_pipeline() function.
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# 1. create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values".
def create_series(arr):
    return pd.Series(arr, name="values")

# 2. clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
def clean_data(series):
    return series.dropna()

# 3. summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
# data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary.
def data_pipeline(arr):
    step1 = create_series(arr)
    step2 = clean_data(step1)
    step3 = summarize_data(step2)
    return step3

# Call data_pipeline(arr) and print each key and its value from the result.
summary_results = data_pipeline(arr)

print("Pipeline Summary:")
for key, value in summary_results.items():
    print(f"{key}: {value:.2f}") 
    