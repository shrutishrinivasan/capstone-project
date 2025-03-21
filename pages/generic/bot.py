import streamlit as st
import json
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import os

# =================================================
# 1. Initialize the Language Model with Ollama - Using Mistral 
# =================================================

model = OllamaLLM(
    model="mistral", 
    base_url="http://localhost:11434",
    temperature=0.1  # Lower temperature for more focused responses
)

# ===================================================
# 2. Document Preprocessing Function
# ===================================================

def docs_preprocessing_helper(file):
    """
    Helper function to load and preprocess a CSV file containing data.
    """
    loader = CSVLoader(file)
    docs = loader.load()
    # Using a smaller chunk size for faster processing
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    return docs

# =====================================================================
# 3. Set up the Embedding Function and Chroma Database
# =====================================================================

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

def setup_chroma_db(docs, embedding_function):
    """Sets up the Chroma database."""
    # Create an in-memory Chroma database
    db = Chroma.from_documents(
        docs, 
        embedding_function,
        persist_directory=None,  # Force in-memory storage
        collection_name="my_collection"  # Give it a specific name
    )
    return db

# =================================================================
# 4. Define and Initialize the Prompt Template (IMPORTANT PART)
# =================================================================

def create_prompt_template():
    """Creates and formats the prompt template."""
    template = """You are a finance consultant chatbot. Answer the customer's questions only using the source data provided.
Please answer to their specific questions. If you are unsure, say "I don't know, please call our customer support". Keep your answers concise.

{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# =====================================================
# 5. Create the Retrieval Chain
# =====================================================

def create_retrieval_chain(model, db, prompt):
    """Creates the retrieval chain."""
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs=chain_type_kwargs,
    )
    return chain

# =================================
# 6. Query the chain and output the response
# =================================

def query_chain(chain, query):
    """Queries the chain and returns the response."""
    response = chain.invoke(query)
    return response['result']

def bot_page():
    """Bot section of the landing page"""

    # Set the font for the entire page and ensure assistant text is white
    st.markdown("""
    <style>
    * {
        font-family: Verdana, sans-serif !important;
    }
    /* Make assistant responses white text */
    [data-testid="stChatMessageContent"] {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Section header with inline horizontal line
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: -20px; margin-top: -10px">
      <h3 style="margin: 0;">Let our bot help you with your queries!</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Only load and process documents once
    if "chain" not in st.session_state:
        with st.spinner("Loading model and data (this will only happen once)..."):
            file_path = r"C:\Users\Rhea Pandita\Desktop\Capstone-Project\final_codes\capstone-project\data\generic2.csv"  # Fixed path with raw string
            docs = docs_preprocessing_helper(file_path)
            db = setup_chroma_db(docs, embedding_function)
            prompt = create_prompt_template()
            st.session_state.chain = create_retrieval_chain(model, db, prompt)

    # Add a button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []  # Clear the messages list
        st.rerun()  # Force Streamlit to rerun the app

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Wrap assistant messages in a div with white text color
            if message["role"] == "assistant":
                st.markdown(f'<div style="color: white;">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("How can I assist you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the query using the LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Implement streaming response feel
            with st.spinner("Thinking..."):
                response = query_chain(st.session_state.chain, prompt)
                # Display response with white text                
                message_placeholder.markdown(f'<div style="color: white;">{response}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})