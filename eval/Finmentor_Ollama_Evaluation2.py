#Finmentor_Ollama_Evaluation2.py
import os, json, pandas as pd, time #standard libraries
import psutil #library for system monitroing such as start, end, cpu, gpu etc

# For NLP/ML evaluation metrics (added from Mistral evaluation)
import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

import torch #for deep learning
from tqdm import tqdm

# Set environment variable to suppress warning messages about HuggingFace symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#Langchain Components
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

def initialize_embedding_function(model_name="sentence-transformers/all-mpnet-base-v2"):
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_function

# Initialize models
model = initialize_model(temperature=0.1)
embedding_function = initialize_embedding_function(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

#=======================================================================================================
# STEP 2: PERFORMANCE MONITORING CLASS 
#=======================================================================================================

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.gpu_available = torch.cuda.is_available()
        self.start_time = None
        self.end_time = None
    
    def start_monitoring(self):
        self.start_time = time.time()
        return {
            "cpu_percent_start": self.process.cpu_percent(interval=0.1),
            "memory_start_mb": self.process.memory_info().rss / (1024 * 1024),
            "gpu_memory_start_mb": torch.cuda.memory_allocated() / (1024 * 1024) if self.gpu_available else 0
        }
    
    def end_monitoring(self):
        self.end_time = time.time()
        return {
            "cpu_percent_end": self.process.cpu_percent(interval=0.1),
            "memory_end_mb": self.process.memory_info().rss / (1024 * 1024),
            "gpu_memory_end_mb": torch.cuda.memory_allocated() / (1024 * 1024) if self.gpu_available else 0,
            "response_time_seconds": self.end_time - self.start_time
        }

# ===================================================
# 3. Document Preprocessing Function
# ===================================================

def docs_preprocessing_helper(file):
    loader = CSVLoader(file)
    docs = loader.load()
    # Using a smaller chunk size for faster processing
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    return docs

# =====================================================================
# 4. Set up the Embedding Function and Chroma Database
# =====================================================================

def setup_chroma_db(docs, embedding_function, persist_directory ):
    """Sets up the Chroma database."""
    db = Chroma.from_documents(docs, embedding_function,
                               persist_directory=persist_directory )
    return db

# =================================================================
# 5. Define and Initialise the Prompt Template (IMPORTANT PART)
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
# 6. Create the Retrieval Chain
# =====================================================

def create_retrieval_chain(model, db, prompt):
    """Creates the retrieval chain."""
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs=chain_type_kwargs
        #return_source_documents=True  # Include source documents in the response
    )
    return chain

# =================================
# 7. Query the chain and output the response
# =================================

#takes a query and runs it through the chain whle monitoring performance
def query_chain(chain, query, performance_monitor):
    """Queries the chain and returns the response with performance metrics."""
    start_metrics = performance_monitor.start_monitoring() # Start monitoring
    response = chain.invoke(query)  # Query the chain
    end_metrics = performance_monitor.end_monitoring() # End monitoring
    performance_metrics = {**start_metrics, **end_metrics} # Combine all metrics
    return response['result'], performance_metrics

# =================================
# 8. Setup and use the QA system
# =================================

'''
def setup_qa_system(csv_path="generic2.csv", model_temperature=0.1, embedding_model_name="sentence-transformers/all-mpnet-base-v2"):
    """Set up the complete QA system."""
    model = initialize_model(temperature=model_temperature)
    embedding_function = initialize_embedding_function(model_name=embedding_model_name)
    docs = docs_preprocessing_helper(csv_path)
    db = setup_chroma_db(docs, embedding_function)
    prompt = create_prompt_template()
    chain = create_retrieval_chain(model, db, prompt)
    return chain
'''
    
#=======================================================================================================
# STEP 9: DEFINE THE TEST CASES
#=======================================================================================================

def create_test_data():
    return [
        {
            "question": "PaisaVault emphasizes the 'butterfly effect' in finance. Could you elaborate on what this means in the context of the app?",
            "reference": "The butterfly effect refers to the idea that small financial choices today, like skipping a daily latte, can have a significant impact on your long-term financial well-being, similar to how a butterfly flapping its wings can supposedly cause a hurricane. PaisaVault uses this concept to show how small daily decisions compound over time."
            #"expected_context": "butterfly effect, financial choices, long-term impact, compounding, decisions"
        },
        {
            "question": "How does PaisaVault utilize AI to help users make better financial decisions?",
            "reference": "PaisaVault incorporates LSTM, a type of artificial intelligence, to analyze past financial data, predict future outcomes, and provide personalized recommendations for improvement. This technology powers features like the Butterfly Effect Simulator and the personalized AI assistant."
            #"expected_context": "LSTM, AI, financial data, predictions, personalized recommendations"
        },
        {
            "question": "What's the primary difference between PaisaVault and other financial apps in terms of data privacy?",
            "reference": "Unlike many finance apps that require linking bank accounts or sharing personal information, PaisaVault prioritizes user privacy. It doesn't collect personally identifiable information (PII) and allows users to manually upload their data, giving them full control."
            #"expected_context": "privacy, PII, manual upload, data control, no bank linking"
        },
        {
            "question": "If I'm a complete beginner with investing, how can PaisaVault assist me?",
            "reference": "PaisaVault offers educational resources and goal-setting tools to help beginners understand basic investment principles. It also simulates how different saving and investment scenarios could affect their long-term financial health. However, it doesn't give specific investment advice."
            #"expected_context": "educational resources, beginners, goal-setting, investment principles, simulations"
        },
        {
            "question": "Explain how the Financial Scenario Tester in PaisaVault can help someone prepare for unforeseen circumstances.",
            "reference": "The Financial Scenario Tester allows users to simulate different financial situations, like a job loss or medical emergency, to assess their financial resilience. This helps them identify potential weaknesses in their financial plan and take proactive steps to strengthen their safety net."
            #"expected_context": "Financial Scenario Tester, resilience, simulations, emergencies, financial plan"
        },
        {
            "question": "How can PaisaVault help someone create and stick to a budget?",
            "reference": "It's recommended to have 3-6 months of essential expenses saved in an easily accessible emergency fund."
            #"expected_context": "budget, expenses, tracking, recommendations, financial habits"
        },
        {
            "question": "What are the key metrics displayed on PaisaVault's dashboard?",
            "reference": "The dashboard visualizes key information like income, expenses, net savings, saving rate, progress towards goals, and financial health scores. It provides a comprehensive overview of your financial situation."
            #"expected_context": "dashboard, metrics, income, expenses, financial health scores"
        },
        {
            "question": "Besides tracking expenses, what other ways does PaisaVault help users improve their financial health?",
            "reference": "Beyond expense tracking, PaisaVault provides personalized advice, simulates the long-term effects of financial decisions, compares spending habits to benchmarks, offers educational resources, and tests financial resilience, all contributing to improved financial habits."
            #"expected_context": "personalized advice, simulations, benchmarks, educational resources, resilience"
        },
        {
            "question": "I'm interested in using PaisaVault but not ready to create an account. What resources are available to me?",
            "reference": "Even without an account, you can access financial calculators, read educational articles, see demos of the app's features, and interact with a generic financial chatbot on PaisaVault's landing page."
            #"expected_context": "no account, calculators, articles, demos, generic chatbot"
        },
        {
            "question": "What is the difference between the generic financial bot and the personalized AI assistant within PaisaVault?",
            "reference": "The generic bot available on the landing page answers general financial questions. The personalized assistant, accessible after creating an account, is trained on your individual financial data to provide tailored insights and recommendations."
            #"expected_context": "generic bot, personalized assistant, account, tailored insights, financial data"
        }
    ] 

#=======================================================================================================
# STEP 10: EVALUATION METRICS IMPLEMENTATION
#=======================================================================================================

# Helper function to extract numerical score from LLM responses
def extract_score(response_text):
    '''
    1. USE REGEX TO FIND A NUMERICAL VALUE IN AN LLM'S RESPONSE TEXT
    2. CONVERT IT TO A FLOAT TYPE
    3. ENSURE THE VALUE IS BETWEEN 1 AND 10
    4. RETURNS A DEFAULT OF 5.0 IF NO NUMBER IS FOUND
    '''
    match = re.search(r'(\d+(?:\.\d+)?)', response_text) # Find a number between 1-10 in the text
    if match:
        score = float(match.group(1))
        return min(max(score, 1), 10) # Ensure score is between 1-10
    # Default fallback
    print(f"WARNING: Could not extract score from: '{response_text}'. Using default score of 5.")
    return 5.0

def evaluate_response_relevancy(question, answer, embedding_model):
    relevancy_prompt = PromptTemplate(
        template="""
        You are evaluating a finance chatbot. Rate how relevant the answer is to the question.
        
        Question: {question}
        Answer: {answer}
        
        Rate ONLY with a number from 1-10 (10 being perfectly relevant).
        """,
        input_variables=["question", "answer"]
    )
    
    try:
        relevancy_chain = relevancy_prompt | embedding_model
        relevancy_response = relevancy_chain.invoke({"question": question, "answer": answer})
        relevancy_score = extract_score(relevancy_response)
        
        # Convert to a 0-1 scale to match other metrics
        return relevancy_score / 10.0
    except Exception as e:
        print(f"Error in response relevancy evaluation: {e}")
        return 0.0

def evaluate_faithfulness(answer, context, evaluator_model):
    faithfulness_prompt = PromptTemplate(
        template="""
        You are evaluating a finance chatbot. Rate if the answer is factually consistent with the context.
        
        Context: {context}
        Answer: {answer}
        
        Rate ONLY with a number from 1-10 (10 being completely faithful to the context).
        """,
        input_variables=["context", "answer"]
    )
    
    try:
        faithfulness_chain = faithfulness_prompt | evaluator_model
        faithfulness_response = faithfulness_chain.invoke({"context": context, "answer": answer})
        faithfulness_score = extract_score(faithfulness_response)
        
        # Convert to a 0-1 scale to match other metrics
        return faithfulness_score / 10.0
    except Exception as e:
        print(f"Error in faithfulness evaluation: {e}")
        return 0.0

def evaluate_semantic_similarity(answer, reference, evaluator_model):
    """
    Evaluates how similar the answer is to the reference using the same approach as relevancy and faithfulness
    """
    reference_similarity_prompt = PromptTemplate(
        template="""
        You are evaluating a finance chatbot. Rate how similar the answer is to the reference answer in terms of content and meaning.
        
        Reference Answer: {reference}
        Chatbot Answer: {answer}
        
        Rate ONLY with a number from 1-10 (10 being very similar to the reference).
        """,
        input_variables=["reference", "answer"]
    )
    
    try:
        reference_chain = reference_similarity_prompt | evaluator_model
        reference_response = reference_chain.invoke({"reference": reference, "answer": answer})
        reference_score = extract_score(reference_response)
        
        # Convert to a 0-1 scale to match other metrics
        return reference_score / 10.0
    except Exception as e:
        print(f"Error in semantic similarity evaluation: {e}")
        return 0.0
    

def calculate_bleu_score(response, reference):
    """
    1. Tokenizes both response and reference into words
    2. Calculates the BLEU score using NLTK's implementation
    3. Uses smoothing to handle cases with no n-gram overlaps
    4. Returns the BLEU score as a measure of text similarity
    5. Includes error handling to return a default of 0.0 if the process fails
    """
    try:
        # Tokenize the response and reference
        response_tokens = nltk.word_tokenize(response.lower())
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        
        # Calculate BLEU score with smoothing
        smoothie = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference_tokens, response_tokens, smoothing_function=smoothie)
        
        return float(bleu_score)
    except Exception as e:
        print(f"Error in BLEU score calculation: {e}")
        return 0.0

def calculate_rouge_scores(response, reference):
    """
    1. Uses the ROUGE library to calculate ROUGE scores
    2. Extracts F1 scores for ROUGE-1, ROUGE-2 and ROUGE-L metrics
    3. Returns all three metrics in a dictionary
    4. Includes error handling to return zeros if the process fails
    """
    try:
        rouge = Rouge()
        scores = rouge.get_scores(response, reference)
        
        # Extract the individual metrics
        rouge_1 = scores[0]['rouge-1']['f']
        rouge_2 = scores[0]['rouge-2']['f']
        rouge_l = scores[0]['rouge-l']['f']
        
        return {
            "rouge_1": float(rouge_1),
            "rouge_2": float(rouge_2),
            "rouge_l": float(rouge_l)
        }
    except Exception as e:
        print(f"Error in ROUGE score calculation: {e}")
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

#=======================================================================================================
# STEP 5: MAIN EVALUATION FUNCTION
#=======================================================================================================

def evaluate_finmentor():
    print("Starting FinMentor Ollama evaluation...")
    
    '''============= STEP 1: INITIALIZE PERFORMANCE MONITOR ===================='''
    performance_monitor = PerformanceMonitor()  # Initialize performance monitor
    test_data = create_test_data()  # Load test data
    print(f"Loaded {len(test_data)} test questions")
    
    '''========== STEP 2: LOAD TEST DATA AND SETUP LANGCHAIN COMPONENTS ========'''   
    # Setup the model chain
    file_path = "C:\\Users\\Rhea Pandita\\Downloads\\capstone-project-main\\data\\custom2.csv"
    docs = docs_preprocessing_helper(file_path)
    persist_dir = "./chroma_finmentor_ollama_eval"
    db = setup_chroma_db(docs, embedding_function, persist_directory=persist_dir)  # create vector database
    prompt = create_prompt_template()
    chain = create_retrieval_chain(model, db, prompt)  # setup prompt template and retrieval chain
    
    # Run evaluation
    results = []
    system_metrics = {
        "average_response_time": 0,
        "average_cpu_usage": 0,
        "average_memory_usage": 0,
        "average_gpu_memory_usage": 0 if torch.cuda.is_available() else None
    }
    
    total_response_times = []
    total_cpu_usage = []
    total_memory_usage = []
    total_gpu_memory_usage = [] if torch.cuda.is_available() else None
    
    '''============== STEP 3: PROCESS EACH TEST QUESTION ========================'''
    for item in tqdm(test_data, desc="Evaluating questions"):
        question = item["question"]
        reference = item["reference"]
        
        # Get context and response
        retriever = db.as_retriever(search_kwargs={"k": 1})
        retrieved_docs = retriever.invoke(question)
        context = retrieved_docs[0].page_content if retrieved_docs else ""
        
        # Get model response with performance metrics
        response, perf_metrics = query_chain(chain, question, performance_monitor)
        
        # Evaluate metrics
        relevancy_score = evaluate_response_relevancy(question, response, model)
        faithfulness_score = evaluate_faithfulness(response, context, model)
        similarity_score = evaluate_semantic_similarity(response, reference, model)
        bleu_score = calculate_bleu_score(response, reference)
        rouge_scores = calculate_rouge_scores(response, reference)
        
        # Track performance metrics
        total_response_times.append(perf_metrics["response_time_seconds"])
        total_cpu_usage.append((perf_metrics["cpu_percent_start"] + perf_metrics["cpu_percent_end"]) / 2)
        memory_usage = (perf_metrics["memory_start_mb"] + perf_metrics["memory_end_mb"]) / 2
        total_memory_usage.append(memory_usage)
        
        if torch.cuda.is_available():
            gpu_memory_usage = (perf_metrics["gpu_memory_start_mb"] + perf_metrics["gpu_memory_end_mb"]) / 2
            total_gpu_memory_usage.append(gpu_memory_usage)
        
        # Store results
        result = {
            "question": question,
            "reference": reference,
            "response": response,
            "context": context,
            "metrics": {
                "response_relevancy": relevancy_score,
                "faithfulness": faithfulness_score,
                "semantic_similarity": similarity_score,
                "bleu_score": bleu_score,
                "rouge_scores": rouge_scores
            },
            "performance": {
                "response_time_seconds": perf_metrics["response_time_seconds"],
                "cpu_percent": (perf_metrics["cpu_percent_start"] + perf_metrics["cpu_percent_end"]) / 2,
                "memory_usage_mb": memory_usage,
                "gpu_memory_usage_mb": gpu_memory_usage if torch.cuda.is_available() else None
            }
        }
        
        results.append(result)
    
    ''' =========== STEP 4: CALCULATE AVERAGE SCORES FOR ALL METRICS =========='''
    # Calculate average scores
    avg_relevancy = np.mean([r["metrics"]["response_relevancy"] for r in results])
    avg_faithfulness = np.mean([r["metrics"]["faithfulness"] for r in results])
    avg_similarity = np.mean([r["metrics"]["semantic_similarity"] for r in results])
    avg_bleu = np.mean([r["metrics"]["bleu_score"] for r in results])
    avg_rouge_1 = np.mean([r["metrics"]["rouge_scores"]["rouge_1"] for r in results])
    avg_rouge_2 = np.mean([r["metrics"]["rouge_scores"]["rouge_2"] for r in results])
    avg_rouge_l = np.mean([r["metrics"]["rouge_scores"]["rouge_l"] for r in results])
    
    # Calculate average performance metrics
    system_metrics["average_response_time"] = np.mean(total_response_times)
    system_metrics["average_cpu_usage"] = np.mean(total_cpu_usage)
    system_metrics["average_memory_usage"] = np.mean(total_memory_usage)
    
    if torch.cuda.is_available():
        system_metrics["average_gpu_memory_usage"] = np.mean(total_gpu_memory_usage)
    
    ''' =================================
    CREATE A SUMMARY CONTAINING:
    1. AVERAGE SCORES FOR ALL EVALUATION METRICS
    2. SYSTEM PERFORMANCE METRICS
    3. INDIVIDUAL RESULTS FOR EACH TEST QUESTION    
    ================================='''
    summary = {
        "average_scores": {
            "response_relevancy": float(avg_relevancy),
            "faithfulness": float(avg_faithfulness),
            "semantic_similarity": float(avg_similarity),
            "bleu_score": float(avg_bleu),
            "rouge_1": float(avg_rouge_1),
            "rouge_2": float(avg_rouge_2),
            "rouge_l": float(avg_rouge_l)
        },
        "system_performance": system_metrics,
        "individual_results": results
    }
    
    # Save results to file
    with open("finmentor_ollama_evaluation2.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nEvaluation complete!")
    print(f"Average Response Relevancy: {avg_relevancy:.4f}")
    print(f"Average Faithfulness: {avg_faithfulness:.4f}")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 Score: {avg_rouge_1:.4f}")
    print(f"Average ROUGE-2 Score: {avg_rouge_2:.4f}")
    print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
    print(f"Average Response Time: {system_metrics['average_response_time']:.4f} seconds")
    print(f"Average CPU Usage: {system_metrics['average_cpu_usage']:.2f}%")
    print(f"Average Memory Usage: {system_metrics['average_memory_usage']:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"Average GPU Memory Usage: {system_metrics['average_gpu_memory_usage']:.2f} MB")
    
    print("Detailed results saved to finmentor_ollama_evaluation2.json")

if __name__ == "__main__":
    evaluate_finmentor()