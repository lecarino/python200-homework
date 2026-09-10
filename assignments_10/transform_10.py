'''

Part 2: Project — The Double-Transform Pipeline

Build transform_10.py, a script that reads from weather_raw, runs the ML classifier and LLM enrichment on each unprocessed record, and writes the results to weather_enriched.

'''

import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Load model and feature list
clf = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )

# ==========================================
# ---Step 1: Incremental Read ---
# ==========================================
print("\n----- Step 1: Incremental Read -----\n")

# Load the model metadata from your models/weather_classifier_metadata.json file. Fetch all rows from weather_raw. Fetch all dates already present in weather_enriched. Determine which records still need processing.
# Print a summary: how many raw records exist, how many are already enriched, and how many will be processed this run.

raw_rows = supabase.table("weather_raw").select("*").execute().data
already_done = {r["date"] for r in supabase.table("weather_enriched").select("date").execute().data}
to_classify = [r for r in raw_rows if r["date"] not in already_done]

print(f"Total raw records: {len(raw_rows)}")
print(f"Already enriched: {len(already_done)}")
print(f"Records to process this run: {len(to_classify)}")

if not to_classify:
    print("Nothing to do — all records already enriched.")
    exit()

# ==========================================
# --- Step 2: ML Transform ---
# ==========================================
print("\n----- Step 2: ML Transform -----\n")


# Load models/weather_classifier.pkl. Build a DataFrame from the unprocessed records, selecting feature columns in the order specified by the metadata. Run predict and predict_proba. 
# Build a list of enrichment records with date, good_for_running, and confidence.
# Print a summary of the predictions: how many days were classified as good, and what is the confidence range?

df = pd.DataFrame(to_classify)
X = df[FEATURES]

predictions   = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]

print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")

enrichment_records = [
    {
        "date":             to_classify[i]["date"],
        "good_for_running": bool(predictions[i]),
        "confidence":       round(float(probabilities[i]), 4),
        "llm_summary":      None,
    }
    for i in range(len(to_classify))
]


# ==========================================
# --- Step 3: LLM Transform ---
# ==========================================

# Design a system prompt and user message function that passes each day's weather features and ML prediction to gpt-4o-mini. 
# For each enrichment record, call the API and add an llm_summary field with the one-sentence recommendation.
# Handle API errors gracefully: use a fallback string rather than crashing. Add a progress print every 50 records.


for i, record in enumerate(enrichment_records):
    raw_row = to_classify[i]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_user_message(
                    raw_row, record["good_for_running"], record["confidence"]
                )},
            ],
            max_tokens=100,
        )
        summary = response.choices[0].message.content.strip()
        record["llm_summary"] = summary or "Recommendation unavailable."
    except Exception as e:
        print(f"  API error on {record['date']}: {e}")
        record["llm_summary"] = "Recommendation unavailable."

    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1} / {len(enrichment_records)}")


# ==========================================
# --- Step 4: Load ---
# ==========================================

# Upsert all enrichment records into weather_enriched. Print the number of rows upserted.

db_response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
print(f"Upserted {len(db_response.data)} rows into weather_enriched")

# ==========================================
# --- Step 5: Verify ---
# ==========================================

# Query weather_enriched and print:
# The total number of rows
# Five sample rows showing date, good_for_running, confidence, and llm_summary
# The number of days classified as good for running
# Add a comment: look at a few of the LLM summaries. Do they accurately reflect the weather features and the model's prediction? Pick one you think is particularly good and one that seems off — what might have caused the weaker one?

total_enriched = supabase.table("weather_enriched").select("date", count="exact").execute()
print(f"\nTotal rows in weather_enriched: {total_enriched.count}")

good_count = (
    supabase.table("weather_enriched")
    .select("date", count="exact")
    .eq("good_for_running", True)
    .execute()
)
print(f"Good-for-running days in weather_enriched: {good_count.count}")

sample = supabase.table("weather_enriched").select("*").limit(5).execute()
print("\nSample Rows:")
for row in sample.data:
    print(f"{row['date']} | good={row['good_for_running']} | conf={row['confidence']:.2f}")
    print(f"  {row['llm_summary']}\n")

# COMMENT (Spot check): Most of the LLM summaries are pretty solid and match the data perfectly. 
# One particularly good one noted the mild temperatures making it a great day for a run. 
# One that seemed slightly off basically just repeated the prediction without adding much flavor, probably because the weather stats were extremely average and didn't trigger any interesting LLM reasoning.

# ==========================================
# --- Step 6: Reflect ---
# ==========================================

# Add a comment block (at least 5-6 sentences) addressing:

# The ML classifier was trained on Charlotte, NC data. If you loaded weather data for a different city in Week 9, do you expect the classifier's predictions to be accurate? Why or why not?
# The LLM recommendations are generated from the model's prediction and the weather features. Does the LLM have any ability to "override" the classifier, or is it purely additive? What are the implications of that?
# If you ran this pipeline on 50,000 records instead of 365, what would be your main concern: cost, latency, or something else? How would you address it?


# COMMENT: 
# If I loaded weather data for a completely different city, I would not expect the classifier's predictions to be accurate. The model learned what constitutes a "good" run based on Charlotte's specific climate, so trying to apply those thresholds to a city with completely different baseline weather patterns would confuse it. 
# The LLM does not have the ability to override the classifier; it is purely additive. It just receives the prediction as a fact and writes a sentence about it, meaning if the ML model makes a bad prediction, the LLM will just confidently explain a wrong answer. 
# If I ran this on 50,000 records, my main concern would be latency. Doing 50,000 API calls sequentially would take hours, since 365 already took my computer a few minutes. I would address it by using concurrency.