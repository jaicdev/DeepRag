from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st # For error reporting in this module if needed
import config # Import config for chunk settings

def load_pdf_documents(file_path):
    """Load documents from a PDF file"""
    try:
        document_loader = PDFPlumberLoader(file_path)
        return document_loader.load()
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return []

def chunk_documents(raw_documents):
    """Split documents into manageable chunks based on config"""
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True # Useful for potential future reference to original location
    )
    return text_processor.split_documents(raw_documents)
