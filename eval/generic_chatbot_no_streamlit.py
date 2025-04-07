import os
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# =================================================
# 1. Initialise the Language Model with Ollama - Using Mistral 
# =================================================

def initialize_model(temperature=0.1):
    """Initialize the LLM model."""
    model = OllamaLLM(
        model="mistral", 
        base_url="http://localhost:11434",
        temperature=temperature  # Lower temperature for more focused responses
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

def initialize_embedding_function(model_name="sentence-transformers/all-mpnet-base-v2"):
    """Initialize the embedding function."""
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_function

def setup_chroma_db(docs, embedding_function):
    """Sets up the Chroma database."""
    db = Chroma.from_documents(docs, embedding_function)
    return db

# =================================================================
# 4. Define and Initialise the Prompt Template (IMPORTANT PART)
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
        return_source_documents=True  # Include source documents in the response
    )
    return chain

# =================================
# 6. Query the chain and output the response
# =================================

def query_chain(chain, query):
    """Queries the chain and returns the response."""
    response = chain.invoke(query)
    return response

# =================================
# 7. Setup and use the QA system
# =================================

def setup_qa_system(csv_path="generic2.csv", model_temperature=0.1, embedding_model_name="sentence-transformers/all-mpnet-base-v2"):
    """Set up the complete QA system."""
    model = initialize_model(temperature=model_temperature)
    embedding_function = initialize_embedding_function(model_name=embedding_model_name)
    docs = docs_preprocessing_helper(csv_path)
    db = setup_chroma_db(docs, embedding_function)
    prompt = create_prompt_template()
    chain = create_retrieval_chain(model, db, prompt)
    return chain

# Simple CLI interface for testing
def main():
    print("Setting up QA system...")
    chain = setup_qa_system()
    print("System ready. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your finance question: ")
        if query.lower() == 'exit':
            break
            
        print("\nProcessing query...")
        response = query_chain(chain, query)
        print("\nAnswer:", response['result'])

if __name__ == "__main__":
    main()