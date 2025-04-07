import os, json, pandas as pd, time #standard libraries
import psutil #library for system monitroing such as start, end, cpu, gpu etc

#for NLP/Ml and evalaution metrics
import numpy as np, nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
nltk.download('punkt', quiet=True) #load Punkt tokenizer model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

from tqdm import tqdm #library for progress display
import torch #for deep-learning

#for creating RAG-Based retrieval QA systems
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#==================================================================================
# STEP 1: ENVIRONEMENT VARIABLES AND MODEL INITIALIZATION
#===================================================================================


from dotenv import load_dotenv # Load environment variables (API keys)
load_dotenv()

# Initialize ChatGroq model with API Key
model = ChatGroq(
    temperature=0.1,
    model_name="mistral-saba-24b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

#For Langchain's use
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#For direct use in evaluation metrics
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


#=================================================================================
# STEP 2: PERFORMANCE MONITORING CLASS
#=================================================================================


# Performance monitoring class
class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.gpu_available = torch.cuda.is_available()
        self.start_time = None
        self.end_time = None
    
    #record start time and initial resource usage (CPU, RAM, GPU, memory)
    def start_monitoring(self):
        """Start monitoring performance metrics"""
        self.start_time = time.time()
        return {
            "cpu_percent_start": self.process.cpu_percent(interval=0.1),
            #convert bytes to MB
            "memory_start_mb": self.process.memory_info().rss / (1024 * 1024), 
            "gpu_memory_start_mb": torch.cuda.memory_allocated() / (1024 * 1024) if self.gpu_available else 0
        }
    
    #record ending resource usage and calculate response time
    def end_monitoring(self):
        """End monitoring and return performance metrics"""
        self.end_time = time.time()
        return {
            "cpu_percent_end": self.process.cpu_percent(interval=0.1),
            "memory_end_mb": self.process.memory_info().rss / (1024 * 1024),
            "gpu_memory_end_mb": torch.cuda.memory_allocated() / (1024 * 1024) if self.gpu_available else 0,
            "response_time_seconds": self.end_time - self.start_time
        }


#===================================================================================
# STEP 3: DOCUMENT PROCESSING HELPER FUNCTIONS
#===================================================================================


'''
Below two functions prepare documents for vector-based retrieval
'''
# Create test data loader functions 
def docs_preprocessing_helper(file):
    loader = CSVLoader(file)
    docs = loader.load()
    #split documents into chunks of 800 characters with no overlap
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    return docs

def setup_chroma_db(docs, embedding_function, persist_directory=None):
    """Sets up the Chroma database with optional persistence directory."""
    if persist_directory and not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    #initialise a chroma vector db from documents using provided embedding func
    db = Chroma.from_documents(docs, embedding_function,
                               persist_directory=persist_directory )
    return db


#===================================================================================
# STEP 4: PROMPT TEMPLATE AND CHAIN CREATION
#===================================================================================


def create_prompt_template():
    template = """
    You are a finance consultant chatbot. 
    Answer the customer's questions only using the source data provided.
    Please answer to their specific questions. 
    If you are unsure, say "I don't know, please call our customer support". 
    Keep your answers concise.

    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

#Builds a Langchain RetrievalQA chain
def create_retrieval_chain(model, db, prompt):
    """Creates the retrieval chain."""
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff", #this method puts all retrieved documents into context
        #fetch top 1 most relevant doc
        retriever=db.as_retriever(search_kwargs={"k": 1}), 
        chain_type_kwargs=chain_type_kwargs,
    )
    return chain

#takes a query and runs it through the chain whle monitoring performance
def query_chain(chain, query, performance_monitor):
    """Queries the chain and returns the response with performance metrics."""
    start_metrics = performance_monitor.start_monitoring() # Start monitoring
    response = chain.invoke(query)  # Query the chain
    end_metrics = performance_monitor.end_monitoring() # End monitoring
    performance_metrics = {**start_metrics, **end_metrics} # Combine all metrics
    return response['result'], performance_metrics


#==================================================================================
# STEP 5: TEST DATA CREATION
#==================================================================================

# evaluation test data
def create_test_data():
    test_data = [
    { "question": "I'm new to budgeting. What's a good starting point?", 
     "reference": "Begin by monitoring your income and expenses.  Note down all your income sources. Then, categorize your spending into essential (rent, bills) and non-essential (dining, entertainment).  Assign budget amounts to each category and regularly track your spending. PaisaVault can simplify this process for you."
     },
    
    { "question": "I've heard of the 50/30/20 budget. How does that actually work?", 
     "reference": "The 50/30/20 rule suggests dividing your income into three main categories: 50% for essential needs like rent and groceries, 30% for wants like entertainment and hobbies, and the remaining 20% for savings and paying off debt. It's a good framework for balancing your spending priorities."
     },
    
    { "question": "What if I can't save 20% of my salary?  What's a realistic savings goal?", 
     "reference": "While 20% is a good target, the important thing is to save consistently, even if it's a smaller amount initially. Start with a percentage you can comfortably manage and gradually increase it as your income rises or expenses decrease.  Even 5-10% is a positive step."
     },
    
    { "question": "Any tips for keeping track of my spending effectively?", 
     "reference": "Categorize all your expenses, regularly review your transactions, and look for patterns. PaisaVault's automated categorization tools can make this easier by sorting your spending, setting category limits, and alerting you when you approach your budget limits. This helps you understand where your money is going and identify potential areas for savings."
     },
    
    { "question": "What are some typical budgeting pitfalls to watch out for?", 
     "reference": "Avoid setting unrealistic budget expectations, remember to include occasional expenses (like insurance premiums), adjust your budget as your situation changes, explore ways to increase income rather than just cutting expenses, and establish an emergency fund for unexpected events. PaisaVault's features can help you steer clear of these common errors."
     },
    
    { "question": "I'm trying to cut back on my spending.  Where should I start?", 
     "reference": "Begin by separating essential needs from discretionary wants.  Review any subscriptions or memberships you rarely use, cut down on dining out, comparison shop for major purchases, and look for energy-saving measures at home.  PaisaVault can analyze your spending and offer personalized suggestions for areas to reduce."
     },
    
    { "question": "What exactly does 'discretionary spending' mean?", 
     "reference": "Discretionary spending refers to the non-essential expenses you can adjust based on your needs, like entertainment, restaurant meals, hobbies, or subscription services. These are distinct from fixed expenses like rent or utilities. They are typically the easiest expenses to reduce when looking to save money."
     },
    
    { "question": "I’m curious about zero-based budgeting.  How does that work?", 
     "reference": "In zero-based budgeting, you allocate every rupee of your income to a specific category (expenses, savings, investments) until your budget reaches zero. This approach forces you to be intentional about all your spending and helps eliminate waste.  PaisaVault's tools can help you implement this method."
     },
    
    { "question": "How much money should I aim to keep in my emergency fund?", 
     "reference": "A good target is 3 to 6 months of essential expenses.  If your income is unstable or you have dependents, aim for 6 to 12 months' worth. Don't feel pressured to save this amount immediately.  Start with a smaller, manageable goal (like ₹25,000-₹50,000) and gradually build it up."
     },
    
    { "question": "What's the best place to keep the money for my emergency fund?", 
     "reference": "Stash your emergency fund in an account that's easily accessible and has minimal risk, like a high-yield savings account or a short-term fixed deposit.  Avoid investments with fluctuating values or withdrawal penalties. The goal is to balance accessibility with earning some interest to offset inflation."
     }
]
    return test_data


#===================================================================================
# STEP 6: EVALUATION METRICS IMPLEMENTATION
#===================================================================================


def evaluate_response_relevancy(user_input, response):
    '''
    1. Uses the LLM to generate questions that the response would answer
    2. Creates embeddings for both the orignal question and generated questions
    3. Calculates cosine similarity between the original question and each generated question.
    4. Returns the average similarity as a measure of relevance
    5. Includes error handling to return 0.0 if the process fails
    '''

    # Generate artificial questions based on the response
    template = f"""
    Based on the following response, generate 3 questions that this response would answer:
    Response: {response}
    
    Output only the questions, one per line, without numbering or any other text.
    """
    
    try:
        questions_result = model.invoke(template)
        generated_questions = questions_result.content.strip().split('\n')
        
        # Remove any empty questions
        generated_questions = [q for q in generated_questions if q.strip()]
        
        if not generated_questions:
            print("No questions were generated.")
            return 0.0
            
        # Compute embeddings for user input and generated questions
        user_embedding = embedding_model.encode([user_input])[0]
        question_embeddings = embedding_model.encode(generated_questions)
        
        # Calculate cosine similarity between user input and each generated question
        similarities = []
        for question_embedding in question_embeddings:
            similarity = cosine_similarity([user_embedding], [question_embedding])[0][0]
            similarities.append(similarity)
        
        # Return the average similarity
        avg_similarity = np.mean(similarities)
        return float(avg_similarity)
    
    except Exception as e:
        print(f"Error in response relevancy evaluation: {e}")
        return 0.0


def evaluate_faithfulness(response, context):
    """
    1. Prompts the LLM to evaluate how faithful the response is to the provided context
    2. Uses regex to extract a numeric score from the LLM's output
    3. Ensures the score is between 0 and 1
    4. Returns a default of 0.5 if the score cant be extracted
    5. Includes error handling o return 0.0 if the process fails
    """
    template = f"""
    Task: Evaluate the faithfulness of a response to the context provided.
    
    Context: {context}
    Response: {response}
    
    Instructions:
    1. Identify all factual claims in the response.
    2. For each claim, determine if it is supported by the context (1) or not (0).
    3. Calculate the proportion of supported claims.
    
    Output only a single number between 0 and 1 representing the faithfulness score, where 1 means completely faithful and 0 means completely unfaithful.
    """
    
    try:
        faithfulness_result = model.invoke(template)
        result_text = faithfulness_result.content.strip()
        
        # Extract the score using a specific pattern
        import re
        score_match = re.search(r'(\d+(\.\d+)?)', result_text)
        
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
        else:
            print(f"Could not extract score from: {result_text}")
            return 0.5
    
    except Exception as e:
        print(f"Error in faithfulness evaluation: {e}")
        return 0.0


def evaluate_semantic_similarity(response, reference):
    """
    1. Creates embeddings for both the model's response and the reference answer
    2. Calculates cosine similairty between these embeddinga
    3. Returns the similairty score as a measure of semantic closeness
    4. Includes error handling to return 0.0 if the process fails
    """
    try:
        # Compute embeddings for response and reference
        response_embedding = embedding_model.encode([response])[0]
        reference_embedding = embedding_model.encode([reference])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([response_embedding], [reference_embedding])[0][0]
        return float(similarity)
    
    except Exception as e:
        print(f"Error in semantic similarity evaluation: {e}")
        return 0.0

def calculate_bleu_score(response, reference):
    """
    1. Tokenizes both response and reference into words
    2. Calculates the BEU score using NLTK's implementation
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

#==================================================================================================================
# STEP 7: MAIN EVALUATION FUNCTION
#=================================================================================================================

# Main evaluation function
def evaluate_finance_chatbot():
    print("Starting Finance Consultant Chatbot evaluation...")
    

    '''============= STEP 1: INITIALIZE PERFORMANCE MONITOR ===================='''
    performance_monitor = PerformanceMonitor() # Initialize performance monitor
    test_data = create_test_data() # Load test data
    print(f"Loaded {len(test_data)} test questions")
    
    '''========== STEP 2: LOAD TEST DATA AND SETUP LANGCHAIN COMPONENTS ========'''
    # Setup the model chain
    file_path = "data/generic2.csv"  
    docs = docs_preprocessing_helper(file_path)
    #persist_dir = "./chroma_generic_eval"
    #db = setup_chroma_db(docs, embedding_function, persist_dir)
    db = setup_chroma_db(docs, embedding_function) #create vector database
    prompt = create_prompt_template()
    chain = create_retrieval_chain(model, db, prompt) #setup prompt template and retrieval chain
    
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
        relevancy_score = evaluate_response_relevancy(question, response)
        faithfulness_score = evaluate_faithfulness(response, context)
        similarity_score = evaluate_semantic_similarity(response, reference)
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
    CREATE A SUMARY CONTAINING:
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
    
    # Save results
    with open("generic_mistral_evaluation.json", "w") as f:
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
    
    print("Detailed results saved to generic_mistral_evaluation.json")


evaluate_finance_chatbot()