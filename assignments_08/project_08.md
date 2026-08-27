# Part 3: Project
Part A: Supabase Setup
Create your Supabase project now so you arrive at Week 9 ready to write code.

# Step 1: Create Your Account and Project
Go to supabase.com and sign up for a free account. No credit card required.
Click New project. Name it python200, choose a region close to you, and set a database password (save it somewhere — you won't need it in this course, but useful to have).
Wait for the project to finish provisioning (usually 30-60 seconds).

# Step 2: Locate Your Credentials
In your project dashboard, click the gear icon (Project Settings) in the left sidebar, then API. You will see:
Project URL — in the form https://<project-id>.supabase.co
anon (public) key — a long string under "Project API keys"
You'll need both of these in Week 9. Keep the tab open for now.

# Step 3: Create a .env File
In a local project folder (or wherever you plan to keep your Week 9 code), create a file called .env:
SUPABASE_URL=https://<your-project-id>.supabase.co
SUPABASE_KEY=<your-anon-key>
Immediately add .env to your .gitignore so it is never committed. API keys committed to a public repository can be scraped within minutes.

# Step 4: Create the Tables
In your Supabase project dashboard, click the SQL Editor icon (</>) in the left sidebar. Run the following SQL to create the two tables you'll use in weeks 9-11:
CREATE TABLE weather_raw (
  date                  date        PRIMARY KEY,
  temperature_2m_max    numeric,
  temperature_2m_min    numeric,
  precipitation_sum     numeric,
  wind_speed_10m_max    numeric,
  loaded_at             timestamptz DEFAULT now()
);

CREATE TABLE weather_enriched (
  date              date        PRIMARY KEY REFERENCES weather_raw(date),
  good_for_running  boolean,
  confidence        numeric,
  llm_summary       text,
  enriched_at       timestamptz DEFAULT now()
);

Then run this to disable Row Level Security on both tables:

ALTER TABLE weather_raw     DISABLE ROW LEVEL SECURITY;
ALTER TABLE weather_enriched DISABLE ROW LEVEL SECURITY;

# Step 5: Confirm
* Project was set up!


# Part B: Cloud Cost Analysis
In project_08.md, write a short summary (a few sentences to a paragraph) covering:
What each scenario costs, and whether the numbers surprised you.
Anything interesting you found while exploring the calculator beyond the two required scenarios.
A sentence on how the two scenarios compare — what does the cost difference tell you about when a GPU instance is or isn't worth it?

* Scenario A — Lightweight compute: cost $2.53 USD per month or $30.36 USD for the whole year. Scenario B — Heavy analytics workload: cost $2,579.65 USD or $30,955.80 USD per year.
* Looking through the calculator, I noticed how fast persistent high-performance GPU instances scale into tens of thousands of dollars, whereas managed object storage (S3) remains remarkably cheap even for large volumes of data.
* The massive cost difference shows that a GPU instance is only worth it if you actively use it for intense, parallelized workloads like model training and shut it down immediately after; leaving a GPU instance running 24/7 idle is a massive waste of money.