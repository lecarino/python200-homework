'''
Part 2: Mini-Project — World Happiness Agent

In Week 1, you built a Prefect pipeline that loaded, cleaned, and analyzed the World Happiness dataset. In this project, you will revisit that same dataset — but this time you will use a CodeAgent to explore it conversationally.
The goal is to build a complete agent from scratch: define the tools, instantiate the agent, and run it through a series of guided queries. Along the way you should see exactly where the agent uses your tools, where it writes its own code, and where its reasoning impresses or surprises you.

Place your code in assignments_07/project_07.py. Save any plots to assignments_07/outputs/.
'''

# ==========================================
# --- Imports ---
# ==========================================
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from smolagents import CodeAgent, OpenAIServerModel, tool

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

# ==========================================
# --- Pre-task: Load the Data ---
# ==========================================
print("\n----- Pre-task: Load the Data -----\n")
DATA_PATH = "../assignments_01/outputs/merged_happiness.csv"

# ==========================================
# --- Task 1: Define Your Tools ---
# ==========================================
print("\n----- Task 1: Define Your Tools -----\n")

df = None

# ===================Tool 1: load_happiness_data===================
@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.
    
    IMPORTANT NOTE FOR AGENT: This tool only returns metadata. If you need to write 
    custom Python code (such as plotting) that requires the full dataset, you must 
    load the data directly inside your generated code using:
    df = pd.read_csv("../assignments_01/outputs/merged_happiness.csv")
    
    Returns:
        A dict with 'shape' (tuple of rows and columns) and 'columns' (list of column names).
    """
    global df
    
    # Try to load the merged file first
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"Successfully loaded merged data from {DATA_PATH}")
    else:
        # Fall back to loading and merging yearly CSVs
        all_data = []
        data_dir = "../resources/happiness_project/" 
        
        for year in range(2015, 2025):
            filepath = os.path.join(data_dir, f"world_happiness_{year}.csv")
            
            if os.path.exists(filepath):
                temp_df = pd.read_csv(filepath, sep=";", decimal=",")
                
                # Standardize columns
                temp_df.columns = temp_df.columns.str.strip().str.lower().str.replace(" ", "_")
                
                # Rename 2024 ladder_score to happiness_score
                if "ladder_score" in temp_df.columns:
                    temp_df = temp_df.rename(columns={"ladder_score": "happiness_score"})
                
                # Add year column
                temp_df["year"] = year
                all_data.append(temp_df)
            else:
                print(f"Warning: File not found for year {year} at {filepath}")
        
        # Merge 
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            print("Successfully merged all yearly CSV files.")
        else:
            return {"error": "No data files were found."}

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


# ===================Tool 2: summarize_column===================
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.
    
    Args:
        column: The name of the column to summarize.
        
    Returns:
        A dictionary of descriptive statistics, or an error dictionary.
    """
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
        
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {df.columns.tolist()}"}
        
    return df[column].describe().to_dict()


# ===================Tool 3: compute_correlation===================
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.
    
    Args:
        col1: The name of the first column.
        col2: The name of the second column.
            
    Returns:
        A dictionary with the correlation coefficient and p-value, or an error dictionary.
    """
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One or both columns not found. Options: {df.columns.tolist()}"}
        
    valid_data = df[[col1, col2]].dropna()
    correlation, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
    
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(correlation, 4),
        "p_value": round(p_value, 4)
    }


# ===================Tool 4: get_top_n_countries===================
@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    
    Args:
        column: The name of the numeric column to sort by.
        year: The year to filter the dataset by.
        n: The number of top countries to return (defaults to 5).
        
    Returns:
        A dictionary containing a list of the top countries and their values, or an error dictionary.
    """
    if df is None:
        return {"error": "No data is loaded yet. Please call load_happiness_data first."}
        
    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available columns: {df.columns.tolist()}"}
        
    if year not in df["year"].values:
        return {"error": f"Year {year} not found in the dataset. Available years: {sorted(df['year'].unique().tolist())}"}
        
    country_col = next((col for col in df.columns if "country" in col.lower()), None)
    if not country_col:
        return {"error": "Could not identify a country column in the dataset."}
        
    filtered_df = df[df["year"] == year].dropna(subset=[column, country_col])
    top_df = filtered_df.sort_values(by=column, ascending=False).head(n)
    
    results_list = top_df[[country_col, column]].rename(columns={country_col: "country"}).to_dict(orient="records")
    
    return {"top_countries": results_list}


# ==========================================
# --- Task 2: Build the Agent ---
# ==========================================
print("\n----- Task 2: Build the Agent -----\n")

model = OpenAIServerModel(model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


# ==========================================
# --- Task 5: Reflection ---
# ==========================================
print("\n----- Task 5: Reflection -----\n")

# --- Reflection ---
#
# 1. In Query 3, how did the agent communicate whether the correlation was statistically
#    significant? Did it use the p-value correctly? What threshold did it apply?
# 
# The agent correctly interpreted the p-value of 0.0. It communicated significance by adding a boolean key `"significant": True`
# to its final output dictionary. While it didn't explicitly state the threshold in the final 
# JSON, by recognizing that 0.0 implies significance, it correctly applied standard statistical 
# thresholds (implicitly p < 0.05 or p < 0.01).
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
# 
# I was surprised by the agent's ability to self correct during code generation 
# in Query 5. When it tried to create the line chart, it encountered a macOS `RuntimeError` 
# because it tried to open an interactive GUI window outside the main thread. Instead of 
# crashing the program, it read the error message, imported `matplotlib`, applied the 
# non-interactive 'Agg' backend (`matplotlib.use('Agg')`), and successfully saved the plot 
# on its second attempt.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
# A `get_country_history(country_name: str)` tool would be useful. It would filter 
# the dataset to return all available metrics for a single country across all years (2015-2024). 
# This would help the agent immediately answer questions like "How has Finland's social support 
# and happiness score changed over the last decade?" without needing to write custom pandas 
# filtering code every time a user asks about a specific country.


if __name__ == "__main__":
    # ==========================================
    # --- Task 3: Run Guided Queries ---
    # ==========================================
    print("\n----- Task 3: Run Guided Queries -----\n")
    os.makedirs("outputs", exist_ok=True)

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # ==========================================
    # --- Task 4: Your Own Questions ---
    # ==========================================
    print("\n----- Task 4: Your Own Questions -----\n")

    # My query 1
    my_query_1 = "What were the top 3 countries for healthy_life_expectancy in 2023?"
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)

    # My query 2
    my_query_2 = "Calculate the average social_support for each region in the year 2022, and create a bar chart showing the results. Save the plot to outputs/social_support_2022.png."
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)

