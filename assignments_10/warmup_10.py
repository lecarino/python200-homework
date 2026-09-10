''' 
Part 1: Warmup
'''

# ========================================== ML vs. LLM in Pipelines ==========================================

# ==========================================
# --- ML/LLM Question 1 ---
# ==========================================
print("\n----- ML/LLM Question 1 -----\n")

# In a comment block, explain the difference between what the ML classifier produces and what the LLM produces in this week's pipeline. 
# Why does each tool do what it does? What would go wrong if you tried to swap them — using the LLM to make the binary good/skip prediction and the ML model to write the recommendation?

# COMMENT: The ML classifier produces a binary prediction, while the LLM produces a short natural-language recommendation. 
# The ML model is used for the prediction because it is faster, cheaper, and more reproducible for structured numeric features. The LLM is used for the recommendation because it can handle language generation and explain things in plain language. 
# If we swapped them, it would fail because the ML model cannot generate plain language explanations, and using an LLM to make the binary prediction would be slower and more expensive than the ML model.


# ==========================================
# --- ML/LLM Question 2 ---
# ==========================================
print("\n----- ML/LLM Question 2 -----\n")

# For each task below, write one sentence in a comment block stating whether you would use a trained ML model, an LLM, or deterministic code, and why:

# Converting a date string like "2023-07-04" to day-of-week
# Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# Predicting customer churn given 15 numeric features and a labeled training dataset
# Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# Summing a column of revenue figures

# COMMENT: 
# deterministic code: date parsing belongs in code and has a single correct answer.
# LLM : reading freeform text requires reading comprehension and judgment that rule-based code handles poorly.
# ML : predicting from a fixed set of numeric features with a well-labeled training set
# LLM : it handles irregular input well for field extraction and normalization.
# deterministic code: arithmetic belongs in code


# ==========================================
# --- ML/LLM Question 3 ---
# ==========================================
print("\n----- ML/LLM Question 3 -----\n")

# In a comment block, answer: what is incremental processing, and why is it important for this pipeline? 
# What would happen — in terms of cost and data correctness — if the transform script re-processed all 365 records every time it ran?

# COMMENT: Incremental processing means the script checks which records are already in the database and only processes the new, unprocessed ones so it doesn't redo completed work. 
# It is important because if the script crashes, you can restart it and it will pick up right where it left off. 
# If the script re-processed all 365 records every time, your API costs would go up because you would pay for 365 LLM calls every single run, and it would be a huge waste of time and compute.



# ========================================== Prompt Design ==========================================


# ==========================================
# --- Prompt Question 1 ---
# ==========================================
print("\n----- Prompt Question 1 -----\n")
# The lesson prompt asks the LLM for exactly one sentence. Write an alternative system prompt that asks for a two-sentence recommendation where the first sentence states the prediction and 
# the second sentence explains the reasoning. In a comment, describe: what would you need to change in the validation logic to accommodate two sentences instead of one?

SYSTEM_PROMPT = (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly two sentences. The first sentence must state the prediction, and the second sentence must explain the reasoning. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

# COMMENT: To accommodate two sentences instead of one, I would need to change the validate_summary logic. 
# Right now, it rejects the text if the length of sentences is greater than 2. 
# I would need to change it to reject if the length does not equal exactly 2, or increase the limit so it doesn't reject valid two-sentence answers.


# ==========================================
# --- Prompt Question 2 ---
# ==========================================

# Write a function call_with_retry(client, messages, max_retries=3) that calls client.chat.completions.create() and retries up to max_retries times on any exception, with a 2-second wait between attempts. 
# On final failure, return None. In a comment, describe when you would use this in a production pipeline.

import time

def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100
            )
            return response
        except Exception as e:
            time.sleep(2)
            
    return None

# COMMENT: I would use this in a production pipeline to handle unexpected network errors or API failures. 
# This makes sure the pipeline is robust and can finish running instead of crashing completely just because one API call timed out.