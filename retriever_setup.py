from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
import streamlit as st # For status updates
import config # Import config for retriever settings
# Import necessary functions and models from llm_utils
from llm_utils import EMBEDDING_MODEL, create_document_summaries, extract_entities

def setup_bm25_retriever(document_chunks):
    """Set up BM25 retriever for keyword-based search"""
    if not document_chunks:
        return None
    try:
        return BM25Retriever.from_documents(
            document_chunks,
            k=config.BM25_K
        )
    except Exception as e:
        st.error(f"Failed to set up BM25 retriever: {e}")
        return None


def setup_multi_vector_retrieval(document_chunks, status_context=None):
    """Set up multi-vector retrieval with original chunks, summaries, and entities"""
    if not document_chunks:
        return None
    if not EMBEDDING_MODEL:
         st.error("Embedding Model not initialized. Cannot set up multi-vector retriever.")
         return None

    # Create the document store for original chunks
    docstore = InMemoryStore()
    id_key = "chunk_id" # Use chunk_id derived from index during creation

    # Add original chunks to docstore and prepare IDs
    doc_ids = []
    for i, doc in enumerate(document_chunks):
        doc_id = str(i)
        doc.metadata[id_key] = doc_id
        doc_ids.append(doc_id)
    docstore.mset([(doc_id, doc) for doc_id, doc in zip(doc_ids, document_chunks)])


    # Initialize vector store for the different representations
    vectorstore = InMemoryVectorStore(embedding=EMBEDDING_MODEL)

    # Create summaries and entity extractions, passing status context
    if status_context: status_context.update(label="Creating document summaries...")
    summaries = create_document_summaries(document_chunks, status_context)

    if status_context: status_context.update(label="Extracting key entities...")
    entities = extract_entities(document_chunks, status_context)

    # Prepare documents for vectorstore embedding (original + summaries + entities)
    # Ensure metadata contains the 'chunk_id' linking back to the original docstore entry
    all_representation_docs = []

    # Add original documents
    all_representation_docs.extend(document_chunks) # Original docs need chunk_id metadata added above

    # Add summaries (ensure they have correct chunk_id metadata)
    all_representation_docs.extend([s for s in summaries if s.metadata.get("chunk_id") in doc_ids])

    # Add entities (ensure they have correct chunk_id metadata and content)
    all_representation_docs.extend([e for e in entities if e.page_content and e.metadata.get("chunk_id") in doc_ids])


    # Add all representations to the vectorstore
    if status_context: status_context.update(label="Embedding document representations...")
    try:
        vectorstore.add_documents(all_representation_docs)
    except Exception as e:
        st.error(f"Failed to add documents to vector store: {e}")
        return None

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        # Search for more documents initially, reranking will narrow down
        search_kwargs={"k": config.MULTI_VECTOR_K * 2} # Fetch more to give reranker options
    )
    return retriever

def setup_hybrid_retriever(multi_vector_retriever, bm25_retriever):
    """Combine dense and sparse retrievers into a hybrid retriever"""
    if not multi_vector_retriever or not bm25_retriever:
        st.warning("One or both base retrievers are missing. Cannot create hybrid retriever.")
        return multi_vector_retriever or bm25_retriever # Return whichever one exists, or None

    return EnsembleRetriever(
        retrievers=[multi_vector_retriever, bm25_retriever],
        weights=config.HYBRID_SEARCH_WEIGHTS
    )
