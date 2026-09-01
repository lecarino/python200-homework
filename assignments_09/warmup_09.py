'''   Part 1: Warmup  '''



# ========================================== Supabase Connection ==========================================


# ==========================================
# --- Connection Question 1 ---
# ==========================================
print("\n----- Connection Question 1 -----\n")

# In a comment block, answer: what are the two pieces of information supabase-py needs to connect to your project?
# Where do you find them in the Supabase dashboard, and why should they never be hardcoded in a Python script?

# COMMENT: Supabase needs the supabase_url that connects to your URL for your project. This is the address your Python script talks to. 
# It also needs the API key (anon key) which is a long string that identifies your application. it is important because it has read and write access to tables by default when Row Level Security is disabled.
# You find them in the Supabase dashboard by going to Project Settings > API. They should never be hardcoded because bots scan public GitHub repositories and can steal your API keys within minutes.


# ==========================================
# --- Connection Question 2 ---
# ==========================================
print("\n----- Connection Question 2 -----\n")

# Write a function get_client() that:
# Loads your credentials from environment variables using python-dotenv
# Creates and returns a Supabase client
# The function should raise a clear error if either environment variable is missing.

import os
from dotenv import load_dotenv
from supabase import create_client

def get_client():
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in the .env file")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase


# ==========================================
# --- Connection Question 3 ---
# ==========================================
print("\n----- Connection Question 3 -----\n")

# In a comment block, answer: what is Row Level Security (RLS), and why did you disable it on your tables for this course? In what kind of real-world application would you want to keep it enabled?

# COMMENT: RLS lets you define fine-grained access policies. We disabled it for this course so that it won't be too complex for us during development. In a real-world application it is important to have RLS so that not everyone can have access in manipulating rows and data.
# For example, in a banking or social media app, RLS ensures a user can only read and edit their own account data, not anyone else's.


# ========================================== supabase-py CRUD ==========================================


# ==========================================
# --- CRUD Question 1---
# ==========================================
print("\n----- CRUD Question 1 -----\n")

# Write a function insert_test_record(supabase) that inserts a single row into weather_raw with today's date and plausible values for all four weather columns. 
# Run it to confirm it works, then add a comment: what would happen if you ran the function twice? How would you change the call to make it safe to run multiple times?

def insert_test_record(supabase):
    # insert single row
    record = {
    "date":               "2026-07-31",
    "temperature_2m_max": 28.4,
    "temperature_2m_min": 17.2,
    "precipitation_sum":  0.0,
    "wind_speed_10m_max": 12.1,
}
    # supabase.table("weather_raw").insert(record).execute()
    supabase.table("weather_raw").upsert(record, on_conflict="date").execute()

    print("Inserted.")

# Run to confirm it works
supabase = get_client()
insert_test_record(supabase=supabase)

# COMMENT: If I ran the function twice it would cause an error because it already has the same primary key. 
# I would change the call to make it safe to run multiple times by making it 'upsert' instead of insert. That way it will make a new row if no existing primary key or will update if there is an existing primary key. 


# ==========================================
# --- CRUD Question 2---
# ==========================================
print("\n----- CRUD Question 2 -----\n")

# Write a function get_records_by_date_range(supabase, start, end) that returns all rows from weather_raw where date >= start and date <= end. The function should return the list of row dictionaries. 
# Test it with a date range that includes the row you inserted in Q1 and print the result.

def get_records_by_date_range(supabase, start, end):
    response = (
        supabase.table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date",end)
        .execute()
    )
    return response.data

records = get_records_by_date_range(supabase, "2026-07-01", "2026-08-31")
print("Fetched records:", records)

# ==========================================
# --- CRUD Question 3---
# ==========================================
print("\n----- CRUD Question 3 -----\n")

# In a comment block, explain the difference between insert and upsert in supabase-py. Give a concrete example of when you would choose each. 
# Then write a function safe_upsert(supabase, records) that upserts a list of records into weather_raw using date as the conflict key and prints the number of rows affected.

# COMMENT: Insert inserts a row into the database table. However, it will cause an error if you try to insert another row with the same primary key. Upsert can insert and update an existing row if there is a row that 
# already has the existing primary key. You could choose insert when initializing data. You could use upsert when updating data. 

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )
    print(f"Rows affected: {len(response.data)}")


# ========================================== Idempotency ==========================================

# ==========================================
# --- Idempotency Question 1---
# ==========================================
print("\n----- Idempotency Question 1 -----\n")

# "Idempotency" means that running an operation multiple times produces the same result as running it once. In a comment block, explain why idempotency matters for a data pipeline.
# Give one concrete example of what goes wrong in a non-idempotent pipeline when the script crashes halfway through and is restarted.

# COMMENT: Idempotency matters for a data pipeline because pipelines often fail and need to be rerun. If a pipeline is idempotent, you can safely restart it without messing up the database.
# For example, if a non-idempotent script tries to insert 100 rows, crashes after 50, and gets restarted, it will insert those first 50 rows a second time, leaving you with duplicate data. An idempotent script (using upsert) would just update the existing 50 and add the rest cleanly.