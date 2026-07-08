import os
import pandas as pd

# Point this to where your class data actually lives
DATA_DIR = "../../python-200/assignments/resources/happiness_project"

# We will store each year's DataFrame in this list
all_data = []

print("--- STARTING THE MERGE PROCESS ---\n")

# Loop through the years
for year in range(2015, 2025):
    filepath = os.path.join(DATA_DIR, f"world_happiness_{year}.csv")
    
    if os.path.exists(filepath):
        print(f"Loading {year}...")
        df = pd.read_csv(filepath, sep=';', decimal=',')
        
        # 1. Look at the raw, messy columns first
        print(f"  Raw Columns: {df.columns.tolist()}")
        
        # 2. Standardize them (lowercase, replace spaces with underscores)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        print(f"  Clean Columns: {df.columns.tolist()}")
        
        # 3. Add the missing year column
        df["year"] = year
        print(f"  Added 'year' column. Data shape is now: {df.shape}\n")
        
        # Add this year's data to our master list
        all_data.append(df)
    else:
        print(f"WARNING: Could not find file for {year}\n")

# 4. Smash them all together!
print("--- SMASHING DATA TOGETHER ---")
merged_df = pd.concat(all_data, ignore_index=True)

# 5. Let's see the final result
print(f"\nFinal Merged Shape: {merged_df.shape}")
print("\nFirst 5 rows of the final dataset:")
print(merged_df.head())


import pandas as pd

print("--- STARTING TASK 2 SANDBOX ---\n")

# 1. Load the merged data we created in Task 1
df = pd.read_csv("outputs/merged_happiness.csv")

# Let's peek at the columns just to be sure what we're working with
print(f"Available Columns: {df.columns.tolist()}\n")

# Figure out the exact name of the happiness column
# (Sometimes it's 'happiness_score', sometimes just 'score')
target_col = "happiness_score" if "happiness_score" in df.columns else "score"
print(f"Using '{target_col}' for our calculations.\n")

# 2. Overall Statistics
print("--- OVERALL STATS ---")
mean_val = df[target_col].mean()
median_val = df[target_col].median()
std_val = df[target_col].std()

print(f"Mean:   {mean_val}")
print(f"Median: {median_val}")
print(f"Std:    {std_val}\n")

# 3. Grouping by Year
print("--- MEAN HAPPINESS BY YEAR ---")
# .groupby() clusters the data by year, then we ask for the mean of just the target column
yearly_mean = df.groupby("year")[target_col].mean()
print(yearly_mean)
print("\n")

# 4. Grouping by Region
print("--- MEAN HAPPINESS BY REGION ---")
# We check if the column actually exists first to avoid crashing
if "region" in df.columns:
    # .sort_values(ascending=False) puts the happiest regions at the top
    regional_mean = df.groupby("region")[target_col].mean().sort_values(ascending=False)
    print(regional_mean)
else:
    print("Uh oh, 'region' column not found. We might need to check the raw data!")