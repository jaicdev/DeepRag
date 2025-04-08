# Configuration settings for the RAG application

# Model Names (using Ollama)
EMBEDDING_MODEL_NAME = "deepseek-r1:1.5b"
LLM_MODEL_NAME = "deepseek-r1:1.5b"

# Text Splitting Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retriever Settings
BM25_K = 5 # Number of results for BM25 retriever
MULTI_VECTOR_K = 5 # Number of results for MultiVector retriever before reranking
HYBRID_SEARCH_WEIGHTS = [0.7, 0.3] # Weights for [MultiVector, BM25]
RERANKED_RESULTS_COUNT = 4 # Final number of documents to pass to LLM

# Prompt Templates
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

RERANKING_PROMPT = """
You are an expert at evaluating document relevance.
Rate how relevant the following document is to the query on a scale of 1-10.
Output only the numeric score.

Query: {query}
Document: {document}

Relevance score (1-10):
"""

ENTITY_EXTRACTION_PROMPT = """
Extract the 5 most important entities (people, organizations, concepts, locations) from this text.
List them separated by commas. If fewer than 5 are prominent, list only those.

Text: {text}

ENTITIES:
"""
