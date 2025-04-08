from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
import streamlit as st # For potential status updates or errors
import config # Import config for model names and prompts

# Initialize models globally within this module
try:
    EMBEDDING_MODEL = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    LANGUAGE_MODEL = OllamaLLM(model=config.LLM_MODEL_NAME)
except Exception as e:
    st.error(f"Failed to initialize Ollama models: {e}. Make sure Ollama is running and the model '{config.LLM_MODEL_NAME}' is available.")
    # Set models to None or raise an exception to prevent further execution
    EMBEDDING_MODEL = None
    LANGUAGE_MODEL = None

def create_document_summaries(document_chunks, status_context=None):
    """Create summaries for each document chunk, updating status label"""
    if not LANGUAGE_MODEL:
        st.error("Language Model not initialized. Cannot create summaries.")
        return []

    summaries = []
    summarize_chain = load_summarize_chain(LANGUAGE_MODEL, chain_type="stuff")
    total_chunks = len(document_chunks)
    batch_size = min(5, total_chunks) # Process in small batches to prevent overwhelming LLM/UI

    for batch_idx in range(0, total_chunks, batch_size):
        end_idx = min(batch_idx + batch_size, total_chunks)
        if status_context:
            status_context.update(label=f"Creating summaries... (Chunk {end_idx}/{total_chunks})")

        for i in range(batch_idx, end_idx):
            doc = document_chunks[i]
            try:
                summary = summarize_chain.run([doc])
                summaries.append(Document(
                    page_content=summary,
                    metadata={"summary_of": doc.metadata.get("source", ""), "chunk_id": str(i)}
                ))
            except Exception as e:
                print(f"Warning: Summarization failed for chunk {i}. Using first sentence. Error: {e}")
                first_sentence = doc.page_content.split('.')[0] + '.'
                summaries.append(Document(
                    page_content=first_sentence,
                    metadata={"summary_of": doc.metadata.get("source", ""), "chunk_id": str(i)}
                ))
    return summaries

def extract_entities(document_chunks, status_context=None):
    """Extract key entities from document chunks, updating status label"""
    if not LANGUAGE_MODEL:
        st.error("Language Model not initialized. Cannot extract entities.")
        return []

    entities = []
    entity_chain = ChatPromptTemplate.from_template(config.ENTITY_EXTRACTION_PROMPT) | LANGUAGE_MODEL
    total_chunks = len(document_chunks)
    batch_size = min(5, total_chunks)

    for batch_idx in range(0, total_chunks, batch_size):
        end_idx = min(batch_idx + batch_size, total_chunks)
        if status_context:
            status_context.update(label=f"Extracting entities... (Chunk {end_idx}/{total_chunks})")

        for i in range(batch_idx, end_idx):
            doc = document_chunks[i]
            try:
                extracted = entity_chain.invoke({"text": doc.page_content})
                entities.append(Document(
                    page_content=extracted,
                    metadata={"entities_from": doc.metadata.get("source", ""), "chunk_id": str(i)}
                ))
            except Exception as e:
                print(f"Warning: Entity extraction failed for chunk {i}. Appending empty. Error: {e}")
                entities.append(Document(
                    page_content="", # Append empty content on failure
                    metadata={"entities_from": doc.metadata.get("source", ""), "chunk_id": str(i)}
                ))
    return entities

def rerank_documents(documents, query):
    """Rerank documents based on relevance to query using LLM"""
    if not LANGUAGE_MODEL:
        st.error("Language Model not initialized. Cannot rerank documents.")
        return documents # Return original order if LLM fails

    reranking_template = ChatPromptTemplate.from_template(config.RERANKING_PROMPT)
    reranking_chain = reranking_template | LANGUAGE_MODEL

    scored_docs = []
    for doc in documents:
        try:
            # Invoke chain to get the score text
            score_text = reranking_chain.invoke({"query": query, "document": doc.page_content})

            # Attempt to parse the score robustly
            try:
                # Extract the last line, split by ':', take the last part, strip whitespace, convert to float
                score = float(score_text.strip().split('\n')[-1].split(':')[-1].strip())
            except (ValueError, IndexError, AttributeError) as parse_error:
                 # Fallback if parsing fails
                 print(f"Warning: Could not parse score from LLM response: '{score_text}'. Error: {parse_error}. Defaulting to 5.0")
                 score = 5.0 # Assign a neutral default score

            scored_docs.append((doc, score))
        except Exception as e:
            print(f"Error reranking document: {e}. Assigning score 0.0")
            scored_docs.append((doc, 0.0)) # Penalize docs that cause errors during reranking

    # Sort by score in descending order
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return only the documents, now reranked
    return [doc for doc, _ in scored_docs]


def generate_answer(user_query, context_documents):
    """Generate answer based on user query and context documents"""
    if not LANGUAGE_MODEL:
        st.error("Language Model not initialized. Cannot generate answer.")
        return "Error: Language Model is unavailable."

    if not context_documents:
        return "I couldn't find relevant information in the document to answer that query."

    # Format context nicely
    context_text = "\n\n---\n\n".join([f"Source Chunk Content:\n{doc.page_content}" for doc in context_documents])

    conversation_prompt = ChatPromptTemplate.from_template(config.PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL

    try:
        with st.spinner("Generating answer..."):
            response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
        return response
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while trying to generate the answer."
