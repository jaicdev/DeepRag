import streamlit as st
import os
import tempfile
from deeprag import process_documents, find_related_documents, generate_answer

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
/* Chat Input Styling */
.stChatInput input {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    border: 1px solid #3A3A3A !important;
}
/* User Message Styling */
.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #1E1E1E !important;
    border: 1px solid #3A3A3A !important;
    color: #E0E0E0 !important;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
/* Assistant Message Styling */
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #2A2A2A !important;
    border: 1px solid #404040 !important;
    color: #F0F0F0 !important;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
h1, h2, h3 {
    color: #00FFAA !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📘 DeepRag")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# Initialize session state
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    if not st.session_state.processed_docs:
        try:
            with st.spinner("Processing document..."):
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_pdf.getbuffer())
                    tmp_path = tmp_file.name
                
                # Process the document using the temporary file path
                process_documents(tmp_path)
                st.session_state.processed_docs = True
                
                # Clean up the temporary file after processing
                os.unlink(tmp_path)
            
            st.success("✅ Document processed successfully with advanced retrieval features!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
    else:
        st.success("✅ Document processed successfully with advanced retrieval features!")
    
    with st.expander("Advanced Retrieval Features"):
        st.markdown("""
        This application uses:
        1. **Hybrid Search** - Combines keyword matching (BM25) with semantic search
        2. **Multi-Vector Retrieval** - Stores multiple representations of each document chunk
        3. **Re-Ranking** - Applies secondary scoring to refine search results
        """)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
            
            with st.expander("View source passages"):
                for i, doc in enumerate(relevant_docs):
                    st.markdown(f"**Passage {i+1}**")
                    st.markdown(doc.page_content)
                    st.divider()
        
        # Force a rerun to update the chat display
        st.rerun()

