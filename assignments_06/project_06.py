'''

Part 2: Mini-Project: Groundwork Coffee Co. Q&A Assistant


Groundwork Coffee Co. is a small, community-focused coffee shop. They have put together a set of documents about their menu, story, hours, loyalty program, and catering services. Your job is to build a RAG-powered assistant that can answer questions about any of those documents accurately.

This is the kind of project you might build in a real job. A business has a folder of internal documents and wants an AI assistant that can answer questions without reading every file manually — and without the AI making things up. By the end of this project, you will have built that system from scratch.

The Groundwork documents are in lessons/06_AI_augmentation/resources/groundwork_docs (link here). Read through them before writing any code. Understanding your data before you start coding will save you time — this is the same principle as the pre-preprocessing step from Week 1.
Place all your code in assignments_06/project_06.py.

'''

# ==========================================
# --- Step 1: Setup ---
# ==========================================
print("\n----- Step 1: Setup -----\n")
# Add your imports at the top of the file. Load your API key from .env and print a confirmation message. Add an assert statement to verify that the groundwork_docs/ directory exists before your code tries to use it.
# An assert statement stops the program early with a clear error message if a condition is not met. For example:
from dotenv import load_dotenv
import os
from pathlib import Path

# ASSERT DIRECTORY EXISTS
docs_dir = Path("../../python-200/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

#LOAD API KEY
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# ==========================================
# --- Step 2: Load the Documents ---
# ==========================================
print("\n----- Step 2: Load the Documents -----\n")
# Load all documents from groundwork_docs/ using SimpleDirectoryReader. Print:
# How many documents were loaded
# The file name of each document
# Hint: each Document object has a metadata dictionary with a "file_name" key.

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader(docs_dir).load_data()

print(f"Amount of documents loaded: {len(docs)}")
for doc in docs:
    print(doc.metadata["file_name"])



# ==========================================
# --- Step 3: Build the Index and Query Engine ---
# ==========================================
print("\n----- Step 3: Build the Index and Query Engine -----\n")

# Build a VectorStoreIndex from the loaded documents and create a query engine with similarity_top_k=3. Print a short confirmation message once the index is ready, such as:
# Index built successfully. Ready to answer questions.

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

#query_engine
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nIndex built successfully. Ready to answer questions.")


# ==========================================
# --- Step 4: Query the Assistant ---
# ==========================================
print("\n----- Step 4: Query the Assistant -----\n")
# Run the five queries below through your query engine. For each one, print:

# The question
# The answer from the model
# The top retrieved source node: document name, similarity score, and the first 200 characters of the chunk text
# Use a loop — do not repeat the same code block five times.

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("\nA:", response)

    top_node = response.source_nodes[0]
    print(f"\nDocument Name: {top_node.node.metadata['file_name']}")
    print(f"Similarity Score: {top_node.score:.4f}")
    print(f"Text Snippet: {top_node.node.get_content()[:200]}...")
    print("-" * 30)
    print('\n')

# After running all five queries, add a comment reflecting on the responses: did the assistant sound confident and accurate? Did any of the answers surprise you?
# COMMENT: The assistant sounded confident and accurate to the questions asked. No answer surprised me at all. 


# ==========================================
# --- Step 5: Find a Failure ---
# ==========================================
print("\n----- Step 5: Find a Failure -----\n")
# Ask the assistant a question you expect it to struggle with. Good candidates include: something vague or ambiguous, something that requires combining information from more than one document, or a question where the answer is simply not in the documents.

# Print the full response and all three retrieved source nodes (document name, similarity score, and first 200 characters of text). Then add a comment explaining:
# What you asked and why you expected it to be hard
# What went wrong — wrong retrieval, missing information, the model guessed anyway?
# When the retrieval failed, did the model's tone change — did it become less certain, or did it still sound confident even when it was wrong? What does this suggest about trusting AI-generated responses?
# What you would change about the system to improve it


struggle_question = "When do dogs need to eat?"
print(f"\nQ: {struggle_question}")
response = query_engine.query(struggle_question)
print("\nA:", response)
for node_with_score in response.source_nodes:
    print(f"\nDocument Name: {node_with_score.node.metadata['file_name']}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
    print("-" * 30)
    print('\n')

# COMMENT: 
# 1. I asked something not related to the model, 'wen do dogs need to eat'. I expected the system to struggle because the answer is not in the provided documents.
# 2. The model retrieved completely irrelevant chunks (FAQ, wholesale, and seasonal menus) with very low similarity scores (~0.70-0.74). Instead of admitting the answer wasn't in the text, it ignored the context and used its general pre-trained knowledge to answer the question anyway.
# 3. The model still sounded highly confident and authoritative, offering standard advice on dog feeding routines. This suggests that you cannot trust an AI's tone or confidence level to determine if it is actually grounding its response in the provided documents.
# 4. What I would change: I would update the system prompt to explicitly enforce strict boundaries: "If the provided context does not contain the answer, you must reply with 'I do not have enough information to answer that based on the provided documents.' Do not use outside knowledge."



# ==========================================
# --- Step 6: Reflection ---
# ==========================================
print("\n----- Step 6: Reflection -----\n")

# Add a comment block at the end of project_06.py answering the following:
# The lesson built semantic RAG manually — chunking, embedding, and indexing took many lines of code. How many lines did the equivalent LlamaIndex implementation take in your project? What does that tell you about the value of using a framework?
# You have now built a system that answers questions from real documents. Describe a different use case — not a coffee shop — where this approach would add genuine value to a business or organization.
# What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?

# COMMENT: 
# 1. The lines equivalent Llamadex implementation took two lines. That tells me the value of frameworks are incredible as it does a lot of things in fewer lines.
# 2. A use case would be a seller-support assistant for digital marketplace platforms. The bot could ingest all of the platform's shipping guidelines, restricted items, and dispute policies. 
# 3. One failure mode RAG cannot fully prevent is a "reasoning failure" or context override. Even if the retriever pulls the perfect, 100% accurate document, the LLM might simply misunderstand the text. Alternatively (as seen in Step 5), the LLM might completely ignore the provided context and confidently hallucinate an answer using its pre-trained knowledge.