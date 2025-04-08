import streamlit as st
import os
import tempfile
import config # Import configuration constants

# Import functions from other modules
from document_processor import load_pdf_documents, chunk_documents
from retriever_setup import setup_bm25_retriever, setup_multi_vector_retrieval, setup_hybrid_retriever
from llm_utils import generate_answer, rerank_documents, LANGUAGE_MODEL, EMBEDDING_MODEL

# --- Session State Initialization ---
# Clear state on first load or if models failed
if "app_ready" not in st.session_state or not LANGUAGE_MODEL or not EMBEDDING_MODEL:
    st.session_state.clear() # Clear all session state
    st.session_state.app_ready = LANGUAGE_MODEL is not None and EMBEDDING_MODEL is not None
    st.session_state.document_chunks = []
    st.session_state.bm25_retriever = None
    st.session_state.multi_vector_retriever = None
    st.session_state.hybrid_retriever = None
    st.session_state.processed_file_name = None


# --- Core Processing Functions ---

def process_documents(file_path):
    """Load, chunk, and set up retrievers for a PDF document."""
    if not st.session_state.app_ready:
         st.error("LLM or Embedding models failed to load. Cannot process documents.")
         return False

    # Use ONE st.status() context for the whole process
    with st.status("Processing document...", expanded=True) as status:
        try:
            status.update(label="Loading PDF document...")
            raw_docs = load_pdf_documents(file_path)
            if not raw_docs:
                status.update(label="Failed to load document.", state="error", expanded=True)
                return False # Stop processing if loading failed

            status.update(label="Chunking document...")
            st.session_state.document_chunks = chunk_documents(raw_docs)
            status.update(label=f"Document split into {len(st.session_state.document_chunks)} chunks.")

            if not st.session_state.document_chunks:
                 status.update(label="No content found after chunking.", state="error", expanded=True)
                 return False

            status.update(label="Setting up keyword search (BM25)...")
            st.session_state.bm25_retriever = setup_bm25_retriever(st.session_state.document_chunks)

            status.update(label="Setting up multi-vector retriever (this may take a moment)...")
            # Pass the 'status' context object down
            st.session_state.multi_vector_retriever = setup_multi_vector_retrieval(
                st.session_state.document_chunks,
                status_context=status
            )

            status.update(label="Configuring hybrid retrieval system...")
            st.session_state.hybrid_retriever = setup_hybrid_retriever(
                st.session_state.multi_vector_retriever,
                st.session_state.bm25_retriever
            )

            if st.session_state.hybrid_retriever:
                status.update(label="Document processing complete!", state="complete")
                st.success("✅ Document processed successfully!") # Add explicit success message outside status
                return True
            else:
                 status.update(label="Failed to set up retrieval system.", state="error", expanded=True)
                 return False

        except Exception as e:
            status.update(label=f"Error during processing: {str(e)}", state="error", expanded=True)
            st.error(f"An error occurred during processing: {str(e)}") # Also show error outside status
            # Clean up potentially partially initialized state
            st.session_state.bm25_retriever = None
            st.session_state.multi_vector_retriever = None
            st.session_state.hybrid_retriever = None
            return False


def find_related_documents(query):
    """Find related documents using hybrid retrieval and reranking."""
    if not st.session_state.get("hybrid_retriever"): # Check if retriever exists
        st.error("Document not processed or retriever not ready. Please upload and process a document first.")
        return []

    # Retrieve documents using hybrid retrieval
    with st.spinner("Retrieving relevant document sections..."):
        try:
            docs = st.session_state.hybrid_retriever.get_relevant_documents(query)
            if not docs:
                st.warning("No relevant sections found by initial retrieval.")
                return []
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            return []

    # Rerank documents
    with st.spinner(f"Reranking {len(docs)} results for better relevance..."):
        try:
            reranked_docs = rerank_documents(docs, query)
        except Exception as e:
            st.error(f"Error during reranking: {e}")
            reranked_docs = docs # Fallback to non-reranked docs if reranking fails

    # Return top N after reranking based on config
    return reranked_docs[:config.RERANKED_RESULTS_COUNT]

# --- Streamlit UI ---

st.set_page_config(layout="wide") # Use wider layout
st.title("DeepRAG")
st.caption(f"Using Ollama ({config.LLM_MODEL_NAME}), Multi-Vector Retrieval, Hybrid Search, and LLM Reranking")

if not st.session_state.app_ready:
    st.error("Initialization failed. Please ensure Ollama is running and models are available, then refresh.")
else:
    uploaded_file = st.file_uploader("1. Upload your PDF document", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        # Process document only if it's new or hasn't been processed successfully
        if st.session_state.processed_file_name != uploaded_file.name:
            st.info(f"Processing '{uploaded_file.name}'...")
            # Use temp file for robust path handling across OS
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                file_path = temp_file.name

            processing_successful = process_documents(file_path)

            # Clean up the temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)

            if processing_successful:
                st.session_state.processed_file_name = uploaded_file.name # Mark file as processed successfully
            else:
                # Reset processed file name if processing failed to allow reprocessing
                st.session_state.processed_file_name = None
                st.session_state.hybrid_retriever = None # Ensure retriever is cleared on failure


    # Only show query section if a document has been successfully processed
    if st.session_state.get("hybrid_retriever") and st.session_state.processed_file_name:
        st.markdown("---")
        st.header(f"2. Ask questions about '{st.session_state.processed_file_name}'")
        user_query = st.text_input("Enter your question:", key="query_input", value="")

        if user_query:
            related_docs = find_related_documents(user_query)

            if related_docs:
                # Generate the answer using the final set of context documents
                answer = generate_answer(user_query, related_docs)

                st.markdown("### Answer")
                st.markdown(answer) # Use markdown for potentially better formatting from LLM

                # Display the context used for the answer
                with st.expander("📚 Show Relevant Context Used for Answer"):
                    for i, doc in enumerate(related_docs):
                        st.markdown(f"**Context Chunk {i+1} (Highly Relevant)**")
                        st.markdown(f"> {doc.page_content}")
                        # Optionally show metadata like page number if available and parsed correctly
                        page_num = doc.metadata.get('page', None)
                        if page_num is not None:
                            st.caption(f"Source: Page {page_num + 1}") # PDFPlumber page numbers are 0-indexed
                        st.markdown("---")
            else:
                # If find_related_documents returned empty (e.g., retrieval or reranking found nothing)
                st.warning("Could not find relevant context to formulate an answer based on the document.")

    elif uploaded_file is not None and st.session_state.processed_file_name != uploaded_file.name:
        st.info("Document processing is required or encountered an error. Check status messages above.")
    elif uploaded_file is None:
         st.info("Upload a PDF document to begin.")
