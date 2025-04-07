import os, json, pandas as pd, time #standard libraries
import psutil #library for system monitroing such as start, end, cpu, gpu etc

#for NLP/Ml and evalaution metrics
import numpy as np, nltk
#from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
nltk.download('punkt', quiet=True) #load Punkt tokenizer model
#from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#from rouge import Rouge

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

# Import RAGAS components
from ragas.metrics import ResponseRelevancy, Faithfulness, SemanticSimilarity, BleuScore, RougeScore
from ragas.metrics import LLMContextPrecisionWithoutReference, LLMContextPrecisionWithReference
from ragas.metrics import LLMContextRecall, NonLLMContextRecall

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper  # Add this import for the wrapper

# Add asyncio to run the async functions
import asyncio

#=============================================================================================
# STEP 1: ENVIRONMENT VARIABLE AND MODEL SETUP
#=============================================================================================

from dotenv import load_dotenv # Load environment variables (API keys)
load_dotenv()

# Initialize models
model = ChatGroq(
    temperature=0.1, #low temperature = more deterministics output, otherwise more randomness
    model_name="mistral-saba-24b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

wrapped_llm = LangchainLLMWrapper(model)  # Create the LangchainLLMWrapper around model

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_function)

#==============================================================================================
# STEP 2: PERFORMANCE MONITORING CLASS
#==============================================================================================
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

#======================================================================================================
# STEP 3: DOCUMENT PROCESSING HELPER FUNCTIONS
#======================================================================================================

''' Below two functions prepare documents for vector-based retrieval '''

def docs_preprocessing_helper(file):
    loader = CSVLoader(file)
    docs = loader.load()
    #split documents into chunks of 800 characters with no overlap
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)
    return docs

def setup_chroma_db(docs, embedding_function, persist_directory):
    """Sets up the Chroma database with a specific persistence directory."""
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    #initialise a chroma vector db from documents using provided embedding func
    db = Chroma.from_documents( docs, embedding_function, persist_directory=persist_directory )
    return db

#=======================================================================================================
# STEP 4: PROMT TEMPLATE AND CHAIN CREATION
#=======================================================================================================

def create_prompt_template():
    """Creates and formats the prompt template."""
    template = """You are a finance consultant chatbot. Answer the customer's questions only using the source data provided.
    Please answer to their specific questions. If you are unsure, say "I don't know, please call our customer support". Keep your answers concise.

    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

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

#Builds a Langchin RetrievalQA chain
def query_chain(chain, query, performance_monitor):
    """Queries the chain and returns the response with performance metrics."""
    start_metrics = performance_monitor.start_monitoring()  # Start monitoring
    response = chain.invoke(query)  # Query the chain
    end_metrics = performance_monitor.end_monitoring() # End monitoring
    performance_metrics = {**start_metrics, **end_metrics} # Combine all metrics
    return response['result'], performance_metrics

#==================================================================================
# STEP 5: TEST DATA CREATION
#==================================================================================

def create_test_data():
    """Create a set of test questions with ground truth answers."""
    # This function would normally load or create a dataset of test Q&A pairs
    # For demo purposes, we'll create a small set inline
    
    test_data = [
        {
            "question": "PaisaVault emphasizes the 'butterfly effect' in finance. Could you elaborate on what this means in the context of the app?",
            "reference": "The butterfly effect refers to the idea that small financial choices today, like skipping a daily latte, can have a significant impact on your long-term financial well-being, similar to how a butterfly flapping its wings can supposedly cause a hurricane. PaisaVault uses this concept to show how small daily decisions compound over time."
        },
        {
            "question": "How does PaisaVault utilize AI to help users make better financial decisions?",
            "reference": "PaisaVault incorporates LSTM, a type of artificial intelligence, to analyze past financial data, predict future outcomes, and provide personalized recommendations for improvement. This technology powers features like the Butterfly Effect Simulator and the personalized AI assistant."
        },
        {
            "question": "What's the primary difference between PaisaVault and other financial apps in terms of data privacy?",
            "reference": "Unlike many finance apps that require linking bank accounts or sharing personal information, PaisaVault prioritizes user privacy. It doesn't collect personally identifiable information (PII) and allows users to manually upload their data, giving them full control."
        },
        {
            "question": "If I'm a complete beginner with investing, how can PaisaVault assist me?",
            "reference": "PaisaVault offers educational resources and goal-setting tools to help beginners understand basic investment principles. It also simulates how different saving and investment scenarios could affect their long-term financial health. However, it doesn't give specific investment advice."
        },
        {
            "question": "Explain how the Financial Scenario Tester in PaisaVault can help someone prepare for unforeseen circumstances.",
            "reference": "The Financial Scenario Tester allows users to simulate different financial situations, like a job loss or medical emergency, to assess their financial resilience. This helps them identify potential weaknesses in their financial plan and take proactive steps to strengthen their safety net."
        },
        {
            "question": "How can PaisaVault help someone create and stick to a budget?",
            "reference": "It's recommended to have 3-6 months of essential expenses saved in an easily accessible emergency fund."
        },
        {
            "question": "What are the key metrics displayed on PaisaVault's dashboard?",
            "reference": "The dashboard visualizes key information like income, expenses, net savings, saving rate, progress towards goals, and financial health scores. It provides a comprehensive overview of your financial situation."
        },
        {
            "question": "Besides tracking expenses, what other ways does PaisaVault help users improve their financial health?",
            "reference": "Beyond expense tracking, PaisaVault provides personalized advice, simulates the long-term effects of financial decisions, compares spending habits to benchmarks, offers educational resources, and tests financial resilience, all contributing to improved financial habits."
        },
        {
            "question": "I'm interested in using PaisaVault but not ready to create an account. What resources are available to me?",
            "reference": "Even without an account, you can access financial calculators, read educational articles, see demos of the app's features, and interact with a generic financial chatbot on PaisaVault's landing page."
        },
        {
            "question": "What is the difference between the generic financial bot and the personalized AI assistant within PaisaVault?",
            "reference": "The generic bot available on the landing page answers general financial questions. The personalized assistant, accessible after creating an account, is trained on your individual financial data to provide tailored insights and recommendations."
        }
    ]
    return test_data

#===================================================================================
# STEP 6: EVALUATION METRICS IMPLEMENTATION
#===================================================================================


#================= METRIC 1: RESPONSE RELEVANCY ==================================
async def evaluate_response_relevancy(question, response, context, evaluator_embeddings, evaluator_llm):
    """
    1. Creates a SingleTurnSample with question, response, and context
    2. Initializes the ResponseRelevancy scorer with LLM and embeddings models
    3. Asynchronously calculates and returns the relevancy score
    4. Includes error handling to return 0.0 if evaluation fails
    """
    try:
        sample = SingleTurnSample(  user_input=question, response=response,
                                    retrieved_contexts=[context] if context else [] )
        
        # Pass both the LLM and embeddings
        scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in response relevancy evaluation: {e}")
        return 0.0

#================# METRIC 2: FAITHFULNESS =======================================
async def evaluate_faithfulness(response, context, evaluator_llm):
    try:
        sample = SingleTurnSample(  user_input="",  # Not used by Faithfulness metric
                                    response=response,
                                    retrieved_contexts=[context] if context else [] )
        
        scorer = Faithfulness(llm=evaluator_llm)
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in faithfulness evaluation: {e}")
        return 0.0

#=================METRIC 3: SEMANTIC SIMILARITY ==================================
async def evaluate_semantic_similarity(response, reference, evaluator_embeddings):
    try:
        sample = SingleTurnSample(  user_input="",  # Not used by SemanticSimilarity metric
                                    response=response,
                                    reference=reference )
        
        scorer = SemanticSimilarity(embeddings=evaluator_embeddings)
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in semantic similarity evaluation: {e}")
        return 0.0

#================= METRIC 4: BLEU SCORE ==========================================
async def calculate_bleu_score(response, reference):
    try:
        sample = SingleTurnSample(  user_input="",  # Not used by BleuScore metric
                                    response=response,
                                    reference=reference )
        
        scorer = BleuScore()
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in BLEU score calculation: {e}")
        return 0.0

#================ METRIC 5: ROUGE SCORES =========================================
async def calculate_rouge_scores(response, reference):
    try:
        sample = SingleTurnSample(  user_input="",  # Not used by RougeScore metric
                                    response=response,
                                    reference=reference )
        
        # Create scorers for each ROUGE type
        rouge1_scorer = RougeScore(rouge_type="rouge1")
        rouge2_scorer = RougeScore(rouge_type="rouge2")
        rougeL_scorer = RougeScore(rouge_type="rougeL")
        
        # Calculate scores for each type
        rouge1_score = await rouge1_scorer.single_turn_ascore(sample)
        rouge2_score = await rouge2_scorer.single_turn_ascore(sample)
        rougeL_score = await rougeL_scorer.single_turn_ascore(sample)
        
        # Extract the individual metrics from the RAGAS score
        return {
            "rouge_1": float(rouge1_score),
            "rouge_2": float(rouge2_score),
            "rouge_l": float(rougeL_score)
        }
    
    except Exception as e:
        print(f"Error in ROUGE score calculation: {e}")
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

#================ METRIC 6: CONTEXT PRECISION WITHOUT REFERENCE ============================
async def evaluate_context_precision_without_reference(question, response, context, evaluator_llm):
    try:
        sample = SingleTurnSample(  user_input=question, response=response,
                                    retrieved_contexts=[context] if context else [] )
        
        scorer = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in context precision (without reference) evaluation: {e}")
        return 0.0

#================ METRIC 7: CONTEXT PRECISION WITH REFERENCE ================================
async def evaluate_context_precision_with_reference(question, response, reference, context, evaluator_llm):
    try:
        sample = SingleTurnSample(  user_input=question, response=response, reference=reference,
                                    retrieved_contexts=[context] if context else [] )
        
        scorer = LLMContextPrecisionWithReference(llm=evaluator_llm)
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in context precision (with reference) evaluation: {e}")
        return 0.0

#================ METRIC 8: LLM -CONTEXT RECALL =================================================
async def evaluate_context_recall(question, response, reference, context, evaluator_llm):
    try:
        sample = SingleTurnSample(  user_input=question, response=response, reference=reference,
                                    retrieved_contexts=[context] if context else [] )
        
        scorer = LLMContextRecall(llm=evaluator_llm)
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in context recall evaluation: {e}")
        return 0.0

#================ METRIC 9: NON-LLM CONTEXT RECALL ===========================================
async def evaluate_non_llm_context_recall(context, reference_contexts, evaluator_embeddings):
    try:
        # For this example, we'll use the reference text to create reference contexts
        # In a real scenario, you'd have actual reference contexts
        if not reference_contexts:
            return 0.0
            
        sample = SingleTurnSample(  retrieved_contexts=[context] if context else [],
                                    reference_contexts=reference_contexts )
        
        scorer = NonLLMContextRecall()
        score = await scorer.single_turn_ascore(sample)
        return float(score)
    
    except Exception as e:
        print(f"Error in non-LLM context recall evaluation: {e}")
        return 0.0

#=======================================================================================================
# STEP 5: MAIN EVALUATION FUNCTIION
#=======================================================================================================
async def evaluate_finmentor():
    print("Starting FinMentor evaluation...")
    
    performance_monitor = PerformanceMonitor() # Initialize performance monitor
    test_data = create_test_data() # Load test data
    print(f"Loaded {len(test_data)} test questions")
    
    # Setup the model chain
    file_path = "C:\\Users\\Rhea Pandita\\Downloads\\capstone-project-main\\data\\custom2.csv"
    docs = docs_preprocessing_helper(file_path)
    persist_dir = "./chroma_finmentor_ragas_eval"
    db = setup_chroma_db(docs, embedding_function, persist_dir)
    prompt = create_prompt_template()
    chain = create_retrieval_chain(model, db, prompt)
    
    '''================  RUN EVALUATION ====================='''
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
    
    '''
    FOR EACH TEST QUESTION:
    1. RETRIEVES RELEVANT CONTEXT FROM THE DATABASE
    2. SPLITS THE REFERENCE INTO SENTENCES FOR REFERENCE CONTEXTS
    3. GETS THE MODEL'S RESPONSE WITH PERFORMANCE MONITORING
    4. APPLIES MULTIPLE EVALAUTION METRCIS USING THE RAGAS FRAMEWORK
    5. EACH EVALAUTION IS RUN ASYNCHRONOUSLY (KEYWOR: AWAIT)
    '''

    for item in tqdm(test_data, desc="Evaluating questions"):
        question = item["question"]
        reference = item["reference"]
        
        # Get context and response
        retriever = db.as_retriever(search_kwargs={"k": 1})
        retrieved_docs = retriever.invoke(question)
        context = retrieved_docs[0].page_content if retrieved_docs else ""
        
        # Breaking down the reference into sentences as a simple way to create reference contexts
        import re
        reference_sentences = re.split(r'(?<=[.!?])\s+', reference)
        reference_contexts = [s for s in reference_sentences if len(s) > 10]  # Filter out very short sentences
        
        # Get model response with performance metrics
        response, perf_metrics = query_chain(chain, question, performance_monitor)
        
        # Evaluate metrics with RAGAS - use wrapped_llm instead of model
        relevancy_score = await evaluate_response_relevancy(question, response, context, evaluator_embeddings, wrapped_llm)
        faithfulness_score = await evaluate_faithfulness(response, context, wrapped_llm)
        similarity_score = await evaluate_semantic_similarity(response, reference, evaluator_embeddings)
        bleu_score = await calculate_bleu_score(response, reference)
        rouge_scores = await calculate_rouge_scores(response, reference)
        
        # Evaluate Context Precision and Recall
        context_precision_wo_ref = await evaluate_context_precision_without_reference(
            question, response, context, wrapped_llm
        )
        context_precision_w_ref = await evaluate_context_precision_with_reference(
            question, response, reference, context, wrapped_llm
        )
        context_recall = await evaluate_context_recall(
            question, response, reference, context, wrapped_llm
        )
        non_llm_context_recall = await evaluate_non_llm_context_recall(
            context, reference_contexts, evaluator_embeddings
        )
        
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
                "rouge_scores": rouge_scores,
                "context_precision_without_reference": context_precision_wo_ref,
                "context_precision_with_reference": context_precision_w_ref,
                "context_recall": context_recall,
                "non_llm_context_recall": non_llm_context_recall
            },
            "performance": {
                "response_time_seconds": perf_metrics["response_time_seconds"],
                "cpu_percent": (perf_metrics["cpu_percent_start"] + perf_metrics["cpu_percent_end"]) / 2,
                "memory_usage_mb": memory_usage,
                "gpu_memory_usage_mb": gpu_memory_usage if torch.cuda.is_available() else None
            }
        }
        
        results.append(result)
    
    '''
    1. CALCULATE AVERAGE SCORES FOR ALL MERICS USING NUMPY
    2. CREATES A STRUCTURED SUMMARY OF ALL RESULTS
    3. SAVES THE DETAILED RESULTS TO JSON FILE
    4. PRINTS THE SUMMARY STATISTICS
    '''
    # Calculate average scores
    avg_relevancy = np.mean([r["metrics"]["response_relevancy"] for r in results])
    avg_faithfulness = np.mean([r["metrics"]["faithfulness"] for r in results])
    avg_similarity = np.mean([r["metrics"]["semantic_similarity"] for r in results])
    avg_bleu = np.mean([r["metrics"]["bleu_score"] for r in results])
    avg_rouge_1 = np.mean([r["metrics"]["rouge_scores"]["rouge_1"] for r in results])
    avg_rouge_2 = np.mean([r["metrics"]["rouge_scores"]["rouge_2"] for r in results])
    avg_rouge_l = np.mean([r["metrics"]["rouge_scores"]["rouge_l"] for r in results])
    
    # Average Context Precision and Recall
    avg_context_precision_wo_ref = np.mean([r["metrics"]["context_precision_without_reference"] for r in results])
    avg_context_precision_w_ref = np.mean([r["metrics"]["context_precision_with_reference"] for r in results])
    avg_context_recall = np.mean([r["metrics"]["context_recall"] for r in results])
    avg_non_llm_context_recall = np.mean([r["metrics"]["non_llm_context_recall"] for r in results])
    
    # Calculate average performance metrics
    system_metrics["average_response_time"] = np.mean(total_response_times)
    system_metrics["average_cpu_usage"] = np.mean(total_cpu_usage)
    system_metrics["average_memory_usage"] = np.mean(total_memory_usage)
    
    if torch.cuda.is_available():
        system_metrics["average_gpu_memory_usage"] = np.mean(total_gpu_memory_usage)
    
    # Prepare summary
    summary = {
        "average_scores": {
            "response_relevancy": float(avg_relevancy),
            "faithfulness": float(avg_faithfulness),
            "semantic_similarity": float(avg_similarity),
            "bleu_score": float(avg_bleu),
            "rouge_1": float(avg_rouge_1),
            "rouge_2": float(avg_rouge_2),
            "rouge_l": float(avg_rouge_l),
            "context_precision_without_reference": float(avg_context_precision_wo_ref),
            "context_precision_with_reference": float(avg_context_precision_w_ref),
            "context_recall": float(avg_context_recall),
            "non_llm_context_recall": float(avg_non_llm_context_recall)
        },
        "system_performance": system_metrics,
        "individual_results": results
    }
    
    # Save results to file
    with open("finmentor_mistral_ragas_evaluation2.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nEvaluation complete!")
    print(f"Average Response Relevancy: {avg_relevancy:.4f}")
    print(f"Average Faithfulness: {avg_faithfulness:.4f}")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 Score: {avg_rouge_1:.4f}")
    print(f"Average ROUGE-2 Score: {avg_rouge_2:.4f}")
    print(f"Average ROUGE-L Score: {avg_rouge_l:.4f}")
    
    # Print Context Precision and Recall
    print(f"Average Context Precision (without reference): {avg_context_precision_wo_ref:.4f}")
    print(f"Average Context Precision (with reference): {avg_context_precision_w_ref:.4f}")
    print(f"Average Context Recall: {avg_context_recall:.4f}")
    print(f"Average Non-LLM Context Recall: {avg_non_llm_context_recall:.4f}")
    
    print(f"Average Response Time: {system_metrics['average_response_time']:.4f} seconds")
    print(f"Average CPU Usage: {system_metrics['average_cpu_usage']:.2f}%")
    print(f"Average Memory Usage: {system_metrics['average_memory_usage']:.2f} MB")
    
    if torch.cuda.is_available():
        print(f"Average GPU Memory Usage: {system_metrics['average_gpu_memory_usage']:.2f} MB")
    
    print("Detailed results saved to finmentor_mistral_ragas_evaluation2.json")


asyncio.run(evaluate_finmentor())