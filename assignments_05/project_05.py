'''Part 2: Mini-Project — Job Application Helper'''


# ==========================================
# --- Task 1: Setup and System Prompt ---
# ==========================================
print("\n-----Task 1: Setup and System Prompt-----\n")
# Task 1: Setup and System Prompt
# Load your API key and initialize the client. Then define a get_completion() helper function (as seen in the prompt engineering lesson) that takes a messages list and returns the model's text response:

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content


# Next, write a system prompt that sets up the model as a job application coach. 
# Be specific: give it a role, a description of who it's helping, and clear behavioral constraints. At a minimum, your system prompt should instruct the model to:

'''
Stay focused on job application materials
Always remind the user to review and edit its output before submitting anywhere
Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment

Before you move on — check: If you print your system prompt and read it aloud, does it sound like a clear briefing for a specific assistant? 
If it's vague or could apply to almost any task, try adding more specificity.
The more concrete your system prompt, the more predictable and useful the model's behavior will be throughout the project.
'''

system_prompt = '''
You are going to be the user's Job Application Helper and Assistant. The user will be a software development student looking for their first technical role. I want you to focus on job application materials. Always remind the user to review and edit the output before submitting anywhere.
Acknowledge that you may not know the user's specific industry norms, and that the user should use their own judgement. You will be the job application coach and take it step-by-step per every detail on what the user should add or leave out
whenever you get a job application input. 
'''

# Add a comment explaining at least one deliberate choice you made in writing the system prompt and why.
# COMMENT: I made sure that the AI helper goes step by step per each detail when looking into the job application materials because i want it to think each step as we know that when the AI does this process, it helps it think more.

print("\n-----Finished Task 1-----\n")



# ==========================================
# --- Task 2: Bullet Point Rewriter ---
# ==========================================
print("\n-----Task 2: Bullet Point Rewriter-----\n")
import json

# Write a standalone rewrite_bullets() function that takes a list of resume bullet points and returns improved versions. This function will later be called from inside the chatbot loop.
'''
Your function should:

Use delimiters to clearly separate the user's bullet points from your instructions
Ask for the output as a JSON list where each item has "original" and "improved" keys
Parse the JSON response and print both versions of each bullet side by side
'''

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Return only the raw JSON object, without any markdown formatting or code blocks. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages=messages)


    try:
        result = json.loads(response)
        
        # Loop through and print side by side
        for i, item in enumerate(result, 1):
            print(f"\n--- Bullet {i} ---")
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}")
            
        return result
        
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        return []

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets=bullets)

# Add a comment: What makes these bullets weak, and what kinds of changes did the model suggest?
# COMMENT: These bullets are vague and not very helpful because there's not a lot of detail. The model suggested adding more detail to the solution, information, and statement.


# ==========================================
# --- Task 3: Cover Letter Generator ---
# ==========================================
print("\n-----Task 3: Cover Letter Generator-----\n")

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    response = get_completion(messages=messages)
    return response

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed a Python course and built data pipelines using Prefect and Pandas."

print(generate_cover_letter(job_title, background))

# Add a comment: Why did you choose those particular examples? What does the few-shot pattern help control in the output?
# COMMENT: These particular examples were given and they're very detail oriented and specific to what I want as an outcome. 
# The few-shot pattern helps control the output in a way that the output is structured in the way the examples were presented. 
# The better the example the more useful and more accurate the output would be to the user's needs. 


# ==========================================
# --- Task 4: Moderation Check ---
# ==========================================
print("\n-----Task 4: Moderation Check-----\n")

# Write an is_safe(text) function that:
# Calls client.moderations.create() with model="omni-moderation-latest"
# Returns True if the input is not flagged, False if it is
# Prints a short, respectful message if the input is flagged, asking the user to rephrase

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged

    if flagged:
        print(f"Please rephrase the input:{text}")
        return False
    else:
        return True

# --- Test Cases ---
# Pass:
safe_text = "I am writing a resume to apply for a software engineering job"
print(f"Testing safe text: '{safe_text}'")
print(f"Result: {is_safe(safe_text)}\n")

# Fail:
unsafe_text = "I am going to kill somebody if i dont get the job!"
print(f"Testing unsafe text: '{unsafe_text}'")
print(f"Result: {is_safe(unsafe_text)}\n")



# ==========================================
# --- Task 5: The Chatbot Loop ---
# ==========================================
print("\n-----Task 5: The Chatbot Loop-----\n")

# Now assemble everything into a working chatbot. Use the starter code below as your structure — your job is to fill in the marked sections.

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            rewritten_bullets = rewrite_bullets(raw_bullets)
            print("\nJob Application Helper: Here are your rewritten bullets:\n")
            print(rewritten_bullets)
            print("\n")
            
            messages.append({"role": "user", "content": f"Can you rewrite these bullets? {raw_bullets}"})
            messages.append({"role": "assistant", "content": rewritten_bullets})
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            cover_letter = generate_cover_letter(job_title,background)
            print("\n--- Here is your Cover Letter Opening ---")
            print(cover_letter)
            print("-----------------------------------------\n")
            # FIXED: Add to history
            messages.append({"role": "user", "content": f"Write a cover letter for {job_title}. Background: {background}"})
            messages.append({"role": "assistant", "content": cover_letter})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            reply = get_completion(messages)
            # - Print the reply
            print(reply)
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": reply})
            pass


if __name__ == "__main__":
    run_chatbot()




# ==========================================
# --- Task 6: Ethics Reflection ---
# ==========================================
print("\n-----Task 6: Ethics Reflection-----\n")

# Option A — Comment block: At the bottom of project_05.py, add a comment block responding to the questions below. Write at least 3-5 sentences total.

# Respond to at least two of the following three questions:
# Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
# What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
# What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)

# 1. Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
#  - If my bot was trained on written text and about certain kinds of people, then the bot will be biased to the opinions or claims of certain specified people. It could favor anything, from communication styles, industries, or cultural backgrounds based on what the bot intakes as its training data or sources. 
# Bots, overall, are trained by whoever or whatever source they train on. Those things are not always objective. If my bot was trained on text written by and about certain kinds of people, it will naturally be biased toward those specific communication styles, industries, or cultural backgrounds.

# 2. What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
#  - Then it's like the bot is doing the whole submission and the user is not at least putting in some effort to correct or adjust the written output from the bot. Many things go wrong like typos or wrong inputs. Bots are not perfect, they may end up putting thins in that's not ideal for the job or employer.
#  So basically if a job-seeker submits the bot's output directly without reviewing it, they risk submitting something with typos, incorrect facts, or a robotic tone that doesn't accurately represent their real experience.

# 3. What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)
#  - If I were deploying this tool professionally, one guardrail I would add is a mandatory UI warning before generating text that explicitly reminds the user to verify all outputs and not misrepresent their actual technical skills to employers.