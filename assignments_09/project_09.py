'''
Part 2: Project — Extract + Load Pipeline 

Build project_09.py, a script that implements a complete Extract + Load pipeline: it fetches 2023 daily weather data from the Open-Meteo API for a city of your choice and loads it into your Supabase weather_raw table.
This is the same data your Week 4 classifier was trained on. In later weeks, you will use these rows as the input to the transform step — so make sure your column values match what the model expects.

'''
# VIDEO LINK : https://youtu.be/11U6lO04bys 

import requests

# ==========================================
# --- Step 1: Extract ---
# ==========================================
print("\n----- Step 1: Extract -----\n")

# Call the Open-Meteo historical archive API to retrieve daily weather data for your chosen city for the full year 2023 (start: 2023-01-01, end: 2023-12-31). Use these four daily variables:
# temperature_2m_max
# temperature_2m_min
# precipitation_sum
# wind_speed_10m_max
# Use response.raise_for_status() to catch errors early. Print a summary of the response once it arrives

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 43.7314,
	"longitude": 7.419,
	"start_date": "2023-01-01",
	"end_date": "2023-12-31",
	"daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
	"timezone": "auto",
}
response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()
print("Successfully fetched weather data from Open-Meteo API.")

# ==========================================
# --- Step 2: Transform ---
# ==========================================
print("\n----- Step 2: Transform -----\n")

# Convert the API response from columnar format into a list of row dictionaries. Each dictionary should have keys that exactly match the column names in weather_raw.
# Print the first and last record to confirm the transformation looks correct. 
# Add a comment: how many records do you expect for a full year, and how many did you get? If the numbers differ, what might explain the discrepancy?

daily = data["daily"]
# print(daily)
records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]

print(f"Prepared {len(records)} records")
print("First record:", records[0])
print("Last record:", records[-1])

# COMMENT: I expect 365 records for a full year since there's 365 days in a year. I got 365 records when i printed out the length of records. 
# If numbers differe then it might've been a leap year or there was a malfunction with recording those specific days missing from 365 days. 


# ==========================================
# --- Step 3: Load ---
# ==========================================
print("\n----- Step 3: Load -----\n")

# Upsert all records into weather_raw. Print a confirmation message showing how many rows were upserted.
# Run the script a second time and confirm the row count in weather_raw does not change. Add a comment: what does this tell you about idempotency?

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

response = (
    supabase.table("weather_raw")
    .upsert(records, on_conflict="date")
    .execute()
)

print(f"Upserted {len(response.data)} rows into weather_raw")

# COMMENT: I ran the script twice and confirmed the rows were 365 both times. 
# This tells me that idempotency is essential for robust pipelines and is effective to run scripts multiple times without having to worry about errors.


# ==========================================
# --- Step 4: Verify ---
# ==========================================
print("\n----- Step 4: Verify -----\n")

# After upserting, run a verification query that:

# Prints the total number of rows in weather_raw
# Prints the earliest and latest dates in the table
# Prints the row for 2023-07-04 (or the nearest date if that date is missing)
# You can also verify directly in the Supabase Table Editor — take a screenshot for the video below.

# Row count
count_response = supabase.table("weather_raw").select("date", count="exact").execute()
print(f"Rows in weather_raw: {count_response.count}")

# Spot-check: first and last record
first = supabase.table("weather_raw").select("*").eq("date", "2023-01-01").execute()
last  = supabase.table("weather_raw").select("*").eq("date", "2023-12-31").execute()
july_4th = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()

print("First record:", first.data)
print("Last record: ", last.data)
print("July 4th record:", july_4th.data)

# Video

# Record a short video (target: 3 minutes, max: 5). Show:

# The script running in your terminal with no errors
# The weather_raw table in your Supabase dashboard with rows visible
# Your verification output printed to the terminal
# Paste the video link in a comment at the top of project_09.py.

