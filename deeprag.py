from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import os
import tempfile

# Constants
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

RERANKING_PROMPT = """
You are an expert at evaluating document relevance.
Rate how relevant each document is to the query on a scale of 1-10.
Consider both keyword matching and semantic relevance.

Query: {query}
Document: {document}

Relevance score (1-10):
"""

# Initialize models and stores
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

# Session state variables
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "multi_vector_retriever" not in st.session_state:
    st.session_state.multi_vector_retriever = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = None

def load_pdf_documents(file_path):
    """Load documents from PDF file"""
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []

def chunk_documents(raw_documents):
    """Split documents into manageable chunks"""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def create_document_summaries(document_chunks):
    """Create summaries for each document chunk"""
    summaries = []
    summarize_chain = load_summarize_chain(LANGUAGE_MODEL, chain_type="stuff")
    
    total_chunks = len(document_chunks)
    # Process in smaller batches to avoid overwhelming the UI
    batch_size = min(5, total_chunks)
    
    for batch_idx in range(0, total_chunks, batch_size):
        end_idx = min(batch_idx + batch_size, total_chunks)
        with st.status(f"Creating summaries... (Batch {batch_idx//batch_size + 1}/{(total_chunks-1)//batch_size + 1})"):
            for i in range(batch_idx, end_idx):
                doc = document_chunks[i]
                try:
                    summary = summarize_chain.run([doc])
                    summaries.append(Document(
                        page_content=summary,
                        metadata={"summary_of": doc.metadata.get("source", ""), "chunk_id": str(i)}
                    ))
                except Exception as e:
                    # Fallback to using first sentence as summary
                    first_sentence = doc.page_content.split('.')[0] + '.'
                    summaries.append(Document(
                        page_content=first_sentence,
                        metadata={"summary_of": doc.metadata.get("source", ""), "chunk_id": str(i)}
                    ))
    return summaries

def extract_entities(document_chunks):
    """Extract key entities from document chunks"""
    entities = []
    entity_extraction_prompt = """
    Extract the 5 most important entities (people, organizations, concepts, locations) from this text:
    {text}
    
    ENTITIES:
    """
    entity_chain = ChatPromptTemplate.from_template(entity_extraction_prompt) | LANGUAGE_MODEL
    
    total_chunks = len(document_chunks)
    # Process in smaller batches
    batch_size = min(5, total_chunks)
    
    for batch_idx in range(0, total_chunks, batch_size):
        end_idx = min(batch_idx + batch_size, total_chunks)
        with st.status(f"Extracting entities... (Batch {batch_idx//batch_size + 1}/{(total_chunks-1)//batch_size + 1})"):
            for i in range(batch_idx, end_idx):
                doc = document_chunks[i]
                try:
                    extracted = entity_chain.invoke({"text": doc.page_content})
                    entities.append(Document(
                        page_content=extracted,
                        metadata={"entities_from": doc.metadata.get("source", ""), "chunk_id": str(i)}
                    ))
                except Exception as e:
                    entities.append(Document(
                        page_content="",
                        metadata={"entities_from": doc.metadata.get("source", ""), "chunk_id": str(i)}
                    ))
    return entities

def setup_multi_vector_retrieval(document_chunks):
    """Set up multi-vector retrieval with original chunks, summaries, and entities"""
    # Create the document store
    docstore = InMemoryStore()
    
    # Initialize vector store for the different representations
    vectorstore = InMemoryVectorStore(embedding=EMBEDDING_MODEL)
    
    # Create summaries and entity extractions
    with st.status("Creating alternative document representations..."):
        st.write("This enhances retrieval by capturing different aspects of your documents")
        summaries = create_document_summaries(document_chunks)
        entities = extract_entities(document_chunks)
    
    # Create the multi-vector retriever
    id_key = "chunk_id"
    
    # Add all documents and their representations to the docstore
    for i, doc in enumerate(document_chunks):
        doc_id = str(i)
        doc.metadata[id_key] = doc_id
        docstore.mset([(doc_id, doc)])
    
    # Create combined list of all vector representations
    all_representations = []
    
    # Add original documents as one representation
    for doc in document_chunks:
        all_representations.append((doc.metadata[id_key], doc))
    
    # Add summaries as another representation
    for summary in summaries:
        if summary.metadata["chunk_id"] in [doc.metadata[id_key] for doc in document_chunks]:
            all_representations.append((summary.metadata["chunk_id"], summary))
    
    # Add entities as another representation
    for entity_doc in entities:
        if entity_doc.metadata["chunk_id"] in [doc.metadata[id_key] for doc in document_chunks]:
            all_representations.append((entity_doc.metadata["chunk_id"], entity_doc))
    
    # Add all representations to the vectorstore
    vectorstore.add_documents([doc for _, doc in all_representations])
    
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs={"k": 5}
    )
    
    return retriever

def setup_bm25_retriever(document_chunks):
    """Set up BM25 retriever for keyword-based search"""
    return BM25Retriever.from_documents(document_chunks)

def setup_hybrid_retriever(multi_vector_retriever, bm25_retriever):
    """Combine dense and sparse retrievers into a hybrid retriever"""
    return EnsembleRetriever(
        retrievers=[multi_vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # Weighting dense retrieval higher than sparse
    )

def rerank_documents(documents, query):
    """Rerank documents based on relevance to query using LLM"""
    reranking_template = ChatPromptTemplate.from_template(RERANKING_PROMPT)
    reranking_chain = reranking_template | LANGUAGE_MODEL
    
    scored_docs = []
    for doc in documents:
        try:
            score_text = reranking_chain.invoke({"query": query, "document": doc.page_content})
            # Extract numeric score from response
            try:
                score = float(score_text.strip())
            except:
                # Fallback if we can't extract a clean number
                score = 5.0
            scored_docs.append((doc, score))
        except Exception as e:
            scored_docs.append((doc, 0.0))
    
    # Sort by score in descending order
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the documents, now reranked
    return [doc for doc, _ in scored_docs]

def process_documents(file_path):
    """Process documents and set up retrievers"""
    with st.status("Processing document...") as status:
        status.update(label="Loading document...")
        raw_docs = load_pdf_documents(file_path)
        
        if not raw_docs:
            raise Exception("Failed to load document. Please check the file format.")
        
        status.update(label="Chunking document...")
        document_chunks = chunk_documents(raw_docs)
        st.session_state.document_chunks = document_chunks
        
        status.update(label="Setting up BM25 retriever...")
        st.session_state.bm25_retriever = setup_bm25_retriever(document_chunks)
        
        status.update(label="Setting up multi-vector retriever...")
        st.session_state.multi_vector_retriever = setup_multi_vector_retrieval(document_chunks)
        
        status.update(label="Configuring hybrid retrieval system...")
        st.session_state.hybrid_retriever = setup_hybrid_retriever(
            st.session_state.multi_vector_retriever,
            st.session_state.bm25_retriever
        )
        
        status.update(label="Document processing complete!", state="complete")

def find_related_documents(query):
    """Find related documents using hybrid retrieval and reranking"""
    # Retrieve documents using hybrid retrieval
    with st.spinner("Retrieving relevant document sections..."):
        docs = st.session_state.hybrid_retriever.get_relevant_documents(query)
    
    # Rerank documents
    with st.spinner("Reranking results for better relevance..."):
        reranked_docs = rerank_documents(docs, query)
    
    return reranked_docs[:4]  # Return top 4 after reranking

def generate_answer(user_query, context_documents):
    """Generate answer based on user query and context documents"""
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

