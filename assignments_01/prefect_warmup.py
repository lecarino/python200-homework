"""
Python 200 - Week 1 Prefect Warmup
"""

import numpy as np
import pandas as pd
from prefect import task, flow

# The starting data
arr_pipe = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task
def create_series(arr):
    return pd.Series(arr, name="values")

@task
def clean_data(series):
    return series.dropna()

@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

@flow(name="data_pipeline")
def pipeline_flow():
    # Prefect tasks are called inside the flow just like normal functions
    series = create_series(arr_pipe)
    cleaned = clean_data(series)
    return summarize_data(cleaned)
    

# This ensures the flow runs when you execute the script from the terminal
if __name__ == "__main__":
    pipeline_flow()


# ==========================================
# Conceptual Questions
# ==========================================

# 1. Why might Prefect be more overhead than it is worth here?
# Answer: For a tiny dataset and simple logic that executes instantly, Prefect introduces unnecessary complexity and execution time overhead. You have to load the Prefect engine, start a flow run, and track state changes, which takes longer than just running the plain Python functions.

# 2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# Answer: 
# - Scheduling: If this pipeline needed to run automatically every morning at 2 AM without human intervention.
# - Error Handling & Retries: If `create_series` pulled data from an unreliable web API instead of a local array, Prefect could automatically retry the task if the connection failed.
# - Logging & Monitoring: Prefect provides a dashboard where you can track exactly how long the flow took, view historical success rates, and set up Slack or email alerts if it fails.
