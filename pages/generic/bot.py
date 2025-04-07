import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# =================================================
# 1. Initialize the Language Model with Groq API
# =================================================

# Load environment variables from .env file
load_dotenv()

def initialize_groq_model():
    """
    Initialize the Groq language model.
    """
    # Retrieve API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Raise error if API key is not found
    if not groq_api_key:
        st.error("Please set your Groq API key in the .env file")
        st.stop()
    
    # Initialize Groq model
    model = ChatGroq(
        groq_api_key=groq_api_key,
        #model_name="llama2-70b-4096",
        model_name="mistral-saba-24b",
        temperature=0.1
    )
    return model

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
# 4. Define and Initialize the Prompt Template
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
            # Replace with your CSV file path
            file_path = "data/generic2.csv"
            
            # Initialize Groq model
            model = initialize_groq_model()
            
            # Preprocess documents
            docs = docs_preprocessing_helper(file_path)
            db = setup_chroma_db(docs, embedding_function)
            prompt = create_prompt_template()
            st.session_state.chain = create_retrieval_chain(model, db, prompt)

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

# Main function to run the app
def main():
    st.title("Finance Consultant Chatbot")
    bot_page()

if __name__ == "__main__":
    main()