'''Part 1: Warmup Exercises'''


# ==========================================
# --- The Chat Completions API ---
# ==========================================
print("\n-----API Question 1-----\n")

# Set up your OpenAI client and make your first chat completion call. 
# Use the model "gpt-4o-mini" and send this prompt: "What is one thing that makes Python a good language for beginners?". Print the model's response.

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

# Print just the text of the response (not the whole object). Then print the name of the model that responded and the total number of tokens used. Label each output.

# Text response:
print("The text of the response:") 
print(response.choices[0].message.content)

# Name of Model: 
print("\nName of the model:")
print(response.model)

# Total num of tokens used: 
print("\nNumber of tokens used:")
print(response.usage.total_tokens)


# ==========================================
# --- API Question 2 ---
# ==========================================
print("\n-----API Question 2-----\n")

# The completions API is stateless — it has no memory of previous calls. The way to give a model context is to pass the conversation history yourself as a list of messages.
# Build the following conversation manually (no loop, no user input — just construct the list) and send it in a single API call:
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

# Print the model's response. Add a comment: Why does the model know Jordan's name, even though it's stateless?
response = client.chat.completions.create(model='gpt-4o-mini',
                                          messages=messages)
print("Model's response for Q2: ")
print(response.choices[0].message.content)

# Comment: The model knows Jordan's name because we passed the entire conversation history into the API call. It has no memory so we have to pass the whole conversation and context in one call. 


# ==========================================
# --- Prompt Engineering ---
# ==========================================
print("\n-----Prompt Question 1 — Zero-Shot-----\n")

#----- get_completion helper function ----
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    """
    Send a prompt to the model and return the assistant's text reply.
    This helper keeps our examples clean and focused on the prompt itself.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content
#----- get_completion helper function ----

# Ask the model to classify the sentiment of each review below as positive, negative, or mixed. Give it no examples — just the task description and the reviews. Print each result labeled with the review number.

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1):
    
    # loop through reviews list and generate prompt for each review
    prompt = f"Classify the sentiment of the following review as positive, negative, or mixed. Review: '{review}'"
    
    # Get the response
    result = get_completion(prompt)
    
    # Print each result labeled with the review number
    print(f"Review {i}: {result}")


# ==========================================
# --- Prompt Engineering Question 2---
# ==========================================
print("\n-----Prompt Question 2 — One-Shot-----\n")

# Repeat the same task, but this time add one example before the reviews to show the model the format you want:

# Example:
example = 'Example: Review: "Fast shipping but the item arrived damaged."  Sentiment: mixed'

for i, review in enumerate(reviews, start=1):
    
    # loop through reviews list and generate prompt for each review
    prompt = f" Classify the sentiment of the following review as positive, negative, or mixed. Here is an example: '{example}'. Here is the Review: '{review}' "
    
    # Get the response
    result = get_completion(prompt)
    
    # Print each result labeled with the review number
    print(f"Review {i}: {result}")
   
# Print the results. Add a comment: Did adding one example change the format or consistency of the output compared to Q1?
# COMMENT: One example changed the format of the output but it didn't change the consistency of the output compared to Q1. 


# ==========================================
# --- Prompt Engineering Question 3---
# ==========================================
print("\n-----Prompt Question 3 — Few-Shot-----\n")

# Repeat the task again, this time with three examples. At least one example should be positive, one negative, and one mixed. 
# Print the results. Add a comment comparing all three approaches (zero-shot, one-shot, few-shot): When would you choose each one?
# Example:
examples = """
Review: "Fast shipping but the item arrived damaged." 
Sentiment: mixed

Review: "The development team built a robust and efficient product." 
Sentiment: positive

Review: "The boss did not treat employees well and rarely gave compliments to the team." 
Sentiment: negative
"""
for i, review in enumerate(reviews, start=1):
    
    # loop through reviews list and generate prompt for each review
    prompt = f" Classify the sentiment of the following review as positive, negative, or mixed. Here are examples: '{examples}'. Here is the Review: '{review}' "
    
    # Get the response
    result = get_completion(prompt)
    
    # Print each result labeled with the review number
    print(f"Review {i}: {result}")

# COMMENT: zero-shot is quicker in a sense that for the user, the user does not have to type in more examples. One-shot is one example to go off of. few-shot is multiple few examples from the user before inputting it into the AI. 
# I would choose zero-shot or one-shot when it comes to simpler things that the AI could possibly already know fully well from data. I would do few-shot when I want to tackle a more complicated subject that could be based on personal opinios or examples. 


# ==========================================
# --- Prompt Engineering Question 4---
# ==========================================
print("\n-----Prompt Question 4 — Chain of Thought-----\n")

# Ask the model to solve the following problem, but instruct it to show its reasoning step by step before giving a final answer. Label the final answer clearly.

problem = '''
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
'''
# Print the full response including the reasoning. Add a comment: Why does asking the model to reason step by step tend to improve accuracy on problems like this?

prompt = f" Solve this problem: {problem}. Show your reasoning step by step before giving a final answer." 
result = get_completion(prompt)
print(result)

# COMMENT: Asking the model to reason step-by-step is good because it forces the model to think and explicitly show its step by step process before concluding to the answer. 


# ==========================================
# --- Prompt Engineering Question 5---
# ==========================================
print("\n-----Prompt Question 5 — Structured Output-----\n")

# Ask the model to analyze the review below and return the result only as valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence). 
# Print the raw response, then parse it with json.loads() and print each field separately, labeled.

import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
prompt = f" Analyze the review: '{review}'. Return only the raw JSON object, without any markdown formatting or code blocks. I expect a valid JSON with keys sentiment, confidence (a float from 0 to 1), and reason (one sentence)." 

# Add a try/except block to handle the case where the response is not valid JSON. If it fails, print the raw response so you can debug the prompt.
response = get_completion(prompt)
print("Raw response:", response)

# Parse JSON safely
try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")


# ==========================================
# --- Prompt Engineering Question 6---
# ==========================================
print("\n-----Prompt Question 6 — Delimiters-----\n")

# Use triple backticks as delimiters to clearly separate the user's text from your instructions. Send the prompt below and print the result.

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = get_completion(prompt)
print("Delimiters Question Response w/ Instructions:", response)

# Then send a second prompt using a passage that is not a set of instructions (any sentence or two of regular prose). 
# Confirm that the model returns "No steps provided." Add a comment: What problem do delimiters help prevent?

user_text2 = "The grass is green and the sky is blue and the sun is yellow."
prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""
response2 = get_completion(prompt2)
print("Delimiters Question Response w/o Instructions:", response2)

# COMMENT: The delimiters help separate the prompt to specific inputs for the Ai to distinguish so it does not get confused with mixing in instructions and/or prompts. 


# ==========================================
# --- Local Models with Ollama ---
# ==========================================
print("\n-----Ollama Question 1-----\n")

# In your terminal, run the following prompt using Ollama (you installed it during the lesson):
"""
Ollama Response (using llama2):
A large language model is a type of artificial intelligence (AI) model that is trained on a vast amount of text data to generate language outputs that are coherent and natural-sounding. These models have become increasingly sophisticated in recent years, with some capable of producing text that is often indistinguishable from human writing, and can be used for a wide range of applications such as chatbots, language translation, and content generation.
"""

prompt = "Explain what a large language model is in two sentences."

response = get_completion(prompt)
print("OpenAI Response:\n", response)

# COMMENT: What differences did you notice between the two responses? 
# this is OPen AI response: A large language model is an artificial intelligence system designed to understand and generate human-like text by analyzing vast amounts of written data. It uses deep learning techniques, particularly neural networks, to predict and produce coherent and contextually relevant language based on the input it receives.
# this is Ollama: A large language model is a type of artificial intelligence (AI) model that is trained on a vast amount of text data to generate language outputs that are coherent and natural-sounding. These models have become increasingly sophisticated in recent years, with some capable of producing text that is often indistinguishable from human writing, and can be used for a wide range of applications such as chatbots, language translation, and content generation.
# The Ollama response is more general in explaining what the LLM is but more open to explaining what they're used for and how long, while OpenAI is more focused on explaining what an LLM is in detail.
# COMMENT: What is one advantage and one disadvantage of running a model locally?
# One advantage is that it's local and does not need any internet services since it's already in your terminal. One disadvantage is that it may be outdated since it wont get updated training data. 