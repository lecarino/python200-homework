'''
Part 1: Warmup Exercises
'''

# ==========================================
# --- Setup ---
# ==========================================

from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ==========================================
# --- RAG Concepts ---
# ==========================================
print("\n-----Concepts Question 1-----\n")

'''

Three teams at a software company are each building a different AI project. Add a comment block to your code that identifies the best approach — prompt engineering, fine-tuning, or RAG — for each scenario, and gives a 1-2 sentence explanation of your reasoning.
Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.


'''

# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
# COMMENT: Since there's a database that the assistant will have to access that's updated every quarter, I believe the best approach here is the RAG approach. It retrieves relevant information from an external database (their internal policy library) then injects it into the LLM assitant.

# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
# COMMENT: Fine tuning would be best for this scenario as it will literally train the neural network to adapt to their specific brand voice. Since there's 3000 examples, it will be perfect for training it into their brand.

# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
# COMMENT: Since it is just a single question, prompt engineering would be best for this scenario as it isn't as complicated as the other two scenarios and the data analyst will only need the work for this specific time. 


# ==========================================
# --- RAG Concepts 2 ---
# ==========================================
print("\n-----Concepts Question 2-----\n")

'''

AI hallucinations (responses that sound confident but are wrong) can be particularly difficult to detect. Add a comment to your code answering this:

Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real situation where a confident hallucination could cause harm.
Think about the tone of the response as well as its content — why does the way the model expresses an answer affect how much we trust it?

'''

# COMMENT: A confidently wrong answer is more harmful because it will make the user think that the AI is correct since the AI is giving the answer without saying it might be false. If the user is heavily dependent on AI then the uuser will be fed wrong information. 
# One real example would be when a student asks AI for answers to a math problem and the AI was confident but gave the wrong answer. Then the student, due to the tone of the response of the AI, will unknowingly trust it and fail for that homework or test. 
# The way the model expresses an answer would make more people trust it because if it's correct most of the time and confident, then the seldom times it answers incorrectly but still confident will just seep into the blind trust of the user. 



# ==========================================
# --- RAG Concepts 3 ---
# ==========================================
print("\n-----Concepts Question 3-----\n")

'''

The steps below make up a complete RAG pipeline, but they are out of order. Copy the list into your code as a comment, arrange them in the correct order, and add a one-sentence description of what happens at each step.

steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]

'''
# ANSWER:

# steps = [
#     "Extract text from source documents",         # Extracting text from external documents 
#     "Split text into chunks",                     # Splitting massive information from external documents into chunks to digest
#     "Convert text chunks into embeddings",        # Run those chunks through an embedding model to turn them into numbers and store them in a database.
#     "Receive the user's query",                   # User types into model
#     "Embed the user's query",                     # Model embeds user's input for LLM
#     "Retrieve the most relevant chunks",          # Model looks at most relevant chunks of input and from chunk embeddings to compare and retrieve relevant stuff
#     "Inject retrieved chunks into the prompt",    # Take those matched chunks of text and secretly paste them into the prompt alongside the user's original question.
#     "Generate a response from the LLM",           # Generate a proper response back to the user
# ]


# ==========================================
# --- Keyword RAG ---
# ==========================================
print("\n-----Keyword RAG-----\n")

# The following questions use the keyword retrieval function from the lesson. Copy the function below into your warmup_06.py — you will call it in the questions that follow.
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# ==========================================
# --- Keyword RAG 1 ---
# ==========================================
print("\n-----Keyword Question 1-----\n")

# Run simple_keyword_retrieval with verbose=True on the query and documents below. Print the name of the selected document.

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

simple_keyword_retrieval(query=query, documents=documents, verbose=True)

# After running the function, add a comment explaining which document was selected and why.
# COMMENT: loyalty.txt was selected because it had an overlap on 'your' but I don't think it should have been the one selected as it doesnt answer the query. 


# ==========================================
# --- Keyword RAG 2 ---
# ==========================================
print("\n-----Keyword Question 2-----\n")

# Run the same function with this second query using the same documents from Q1:

query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query=query,documents=documents,verbose=True)

# Add a comment explaining:

# Which document was selected
# Whether keyword RAG got this right — and why or why not
# What kind of retrieval would do better here

# COMMENT: Which document was selected? None. Because there were no overlaps found. Whether keyword RAG got this right — and why or why not? keyword RAG did not get it right becuase none of the keywords were triggered to answer the query.
# What kind of retrieval would do better here? naive semantic RAG would be better as it is more complex and takes chunks of text then embeds it and stores in a vector database to use. 


# ==========================================
# --- Keyword RAG 3 ---
# ==========================================
print("\n-----Keyword Question 3-----\n")

# Before running any code, predict which document will be selected for the query below. Write your prediction and your reasoning as a comment first, then run the code to check.
query = "How do I sign up for rewards?"
# COMMENT: PREDICTION -> It won't pick anything because there are no keywords to match with the documents. 

simple_keyword_retrieval(query=query,documents=documents,verbose=True)

# Was your prediction correct? If the result surprised you, add a comment explaining what happened.
# COMMENT: My prediction was correct and it was because there were no overlapping keywords. 



# ==========================================
# --- Semantic RAG Concepts ---
# ==========================================
print("\n-----Semantic Question 1-----\n")

# Add a comment block answering the following in your own words. Try not to just copy the definitions from the lesson — explaining a concept in your own words is a good sign that you have understood it.

# 1. What is a vector embedding? (1-2 sentences)
    #  Vector embedding is a vector of numbers that captures the meaning of something. Similar meanings equals similar vectors
# 2. Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is more relevant, and what does that number tell you about the relationship between the texts?
    # 0.85 score is more relevant because it means that it's closer to that relationship between the texts. the higher the number the higher the relevance
# 3. Why can semantic search find a relevant chunk even when none of the exact words from the query appear in the chunk?
    # Semantic search finds relevant chunks because it looks at meanings of words and not excat words. Similar or close enough meanings will be retrieved. 


# ==========================================
# --- Semantic RAG Concepts 2---
# ==========================================
print("\n-----Semantic Question 2-----\n")

# Keyword RAG and semantic RAG handle the same problem differently. Copy this table into your code as a comment and fill in the right column:

# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Meaning      |
# | What is retrieved?         | Full document                     | Relevant Chunks |
# | Can it handle synonyms?    | No                                | Yes            |
# | Storage format             | Plain text dictionary             | Vector dataspaces            |
# | Relevance score            | Number of overlapping keywords    | cosine similarity            |


# ==========================================
# --- LlamaIndex---
# ==========================================
print("\n-----LLAMA Index-----\n")

# For this section you will build a small LlamaIndex pipeline using the Brightleaf Solar PDFs from the lesson. These documents should already be familiar from the lesson material.
# Path note: The brightleaf_pdfs/ directory is in the lesson folder, not the assignments folder. Point SimpleDirectoryReader to it using a path relative to where you run your script — for example:
# SimpleDirectoryReader("../../06_AI_augmentation/brightleaf_pdfs")
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

file_path = '../../python-200/lessons/06_AI_augmentation/resources/brightleaf_pdfs'

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader(file_path).load_data()

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

# Adjust this path as needed based on your local folder structure.
# API note: These questions make a small number of calls to the OpenAI embeddings API to build the vector index. The cost is very low (typically less than one cent), but make sure your .env file has a valid key before running.


# ==========================================
# --- LlamaIndex 1---
# ==========================================
print("\n-----LlamaIndex Question 1-----\n")

# Build an in-memory LlamaIndex pipeline using the Brightleaf Solar PDFs and run the two queries below. For each query, print:
# The question
# The answer from the model
# For each of the 3 retrieved source nodes: the similarity score and the first 150 characters of the chunk text

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

# Use similarity_top_k=3. After printing the results, add a comment for each query answering:
# Do the retrieved chunks look relevant to the question?
# Does the model's response sound confident and specific, or does it hedge with phrases like "based on the context" or "I'm not sure"? Note what you observe about the tone.
# Did anything unexpected get retrieved?

query_engine = index.as_query_engine(similarity_top_k=3)

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("\nA:", response)
    
    for node_with_score in response.source_nodes:
        # print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)
        print('\n')
# Query 1 ("What employee benefits does BrightLeaf offer?"):
# - Relevance: The top node (Score: 0.9075) was extremely relevant, pulling directly from the well-being and benefits section.
# - Tone: Highly confident, detailed, and direct. It listed specific perks (401k match, health insurance, parental leave) without hedging or saying "based on the context."
# - Unexpected Retrievals: Node 3 (Score: 0.8115) was pulled from "Network and Data Security". Because similarity_top_k=3 forces LlamaIndex to return 3 chunks, it retrieved a security chunk even though it was unrelated to employee benefits.

# Query 2 ("What are BrightLeaf's security policies?"):
# - Relevance: The top node (Score: 0.8797) was directly from the Network and Data Security document and contained the exact policies needed.
# - Tone: Authoritative and specific, referencing technical standards like TLS 1.3, AES-256 encryption, and NIST 800-61 guidance.
# - Unexpected Retrievals: Nodes 2 and 3 pulled from the Employee Well-being and Company Overview documents (scores ~0.82–0.84) simply to satisfy the top_k=3 requirement, even though Node 1 already had all the security information.

# ==========================================
# --- LlamaIndex 2---
# ==========================================
print("\n-----LlamaIndex Question 2-----\n")

# Re-run one of the queries from Q1 twice: once with similarity_top_k=1 and once with similarity_top_k=5. Print the response and source node scores for both runs.

#------ similarity_top_k=1 ------
q1 = questions[0]
query_engine_k1 = index.as_query_engine(similarity_top_k=1)

print(f"\nQ: {q1}")
response = query_engine_k1.query(q1)
print("\nA:", response)
    
for node_with_score in response.source_nodes:
    # print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)
    print('\n')

#------ similarity_top_k=5 ------
query_engine_k5 = index.as_query_engine(similarity_top_k=5)

print(f"\nQ: {q1}")
response = query_engine_k5.query(q1)
print("\nA:", response)

for node_with_score in response.source_nodes:
    # print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)
    print('\n')

# Add a comment explaining how the response changed (if at all) and whether more retrieved context is always better.

# COMMENT: 
# When using top_k=1, the model retrieved a single, highly relevant chunk (Score 0.9075) and used it to provide a concise and accurate answer detailing the benefits program.
# When using top_k=5, the model retrieved the exact same highly relevant chunk, but it also pulled in four additional chunks (with scores ranging from 0.7903 to 0.8147) detailing the company overview, network security, a 2022 EcoVolt partnership, and a financial performance report.
# More retrieved context is NOT always better. While the model still generated a good response, forcing a higher top_k pulls in irrelevant "noise" (like security policies and financial performance) just to hit the quota. This wastes tokens, increases costs, and raises the risk of the model getting confused or hallucinating.


# ==========================================
# --- LlamaIndex 3---
# ==========================================
print("\n-----LlamaIndex Question 3-----\n")

# Try a query you think the pipeline might struggle with — something vague, something that spans multiple documents, or something where the information might not be in the documents at all. Print the response and all retrieved chunks.

personal_query = "Why is the sky blue and the grass green?"
query_engine = index.as_query_engine()

print(f"\nQ: {personal_query}")
response = query_engine.query(personal_query)
print(f"\nA: {response}" )
for node_with_score in response.source_nodes:
    # print(f"Node ID: {node_with_score.node.node_id}")
    print(f"\nSimilarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)
    print('\n')


# Add a comment explaining what you expected, what actually happened, and what you would change about the system to handle this kind of query better.
# COMMENT: I expected the system to fail or say "I don't know" because the sky/grass information is not in the BrightLeaf PDFs. 
# What actually happened was the LLM ignored the retrieved chunks and answered using its general pre-trained knowledge. 
# To handle this better, I would change the system prompt to explicitly enforce: "If the answer is not contained in the provided context, you must respond with 'I do not have enough information to answer that.' Do not use outside knowledge."
# Because it could be hallucination. 

# ==========================================
# --- LlamaIndex 4---
# ==========================================
print("\n-----LlamaIndex Question 4-----\n")

# Using the same index and query engine you built in Q1, evaluate one response using LlamaIndex's built-in evaluators.
query_engine = index.as_query_engine(similarity_top_k=3)

#Import and instantiate a FaithfulnessEvaluator and a RelevancyEvaluator, both using gpt-4o-mini as the judge LLM 
# (refer to the "RAG Evaluation using LlamaIndex" section of lesson 4 for the exact import and setup pattern). Run them on this query:
q = "What employee benefits does BrightLeaf offer?"

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Get response to query
response = query_engine.query(q)

# Print both scores. Then run the evaluators again on a query you expect to produce a lower-quality response — for example, a question about something that is clearly not in the Brightleaf documents.

#############INITIAL QUERY##############
# Evaluate faithfulness and relevancy
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

#############LOWER-QUALITY-QUERY##############
q2 = "What does a cybersecurity attack look like?"
response2 = query_engine.query(q2)

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q2, response=response2)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q2, response=response2)
print("Relevancy Result: " + str(relevancy_result.score))

# After printing both sets of scores, add a comment block answering:

# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
# What does a relevancy score measure, and how is it different from faithfulness?
# Did the scores change between your two queries? If so, why do you think that happened?
# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?

# COMMENT: 
# Faithfulness of 1.0 means the answer is grounded in the retrieved chunks without hallucinating outside details. A 0.0 means the answer contains claims not from the documents the user input.
# Relevancy measures whether the answer addresses the user's prompt. It's different from faithfulness because the answer can be correct and not be from the retrieved documents.
# Yes, the faithfulness changed because the first query was relevant while the second query wasn't relevant to the documents at all.
# The "LLM-as-a-judge" approach uses an advanced LLM (like gpt-4o-mini) to evaluate the semantic quality, factual grounding, and intent alignment of a response. It is used for RAG evaluation because simple metrics (like keyword matching or string overlap) cannot assess whether an answer logically follows from context or properly answers a open-ended question.
