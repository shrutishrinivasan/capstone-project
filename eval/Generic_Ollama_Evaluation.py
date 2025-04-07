import os
import json
import time
import re
import pandas as pd
import psutil #for monitoring system resources (CPU, memory)
import tracemalloc #for tracking memory allocations
import traceback #for handling exceptions
from datetime import datetime

# Set environment variable to suppress warning messages about HuggingFace symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Import required functions from the new isolated code file
from generic_chatbot_no_streamlit import initialize_model, initialize_embedding_function, docs_preprocessing_helper
from generic_chatbot_no_streamlit import setup_chroma_db, create_prompt_template, create_retrieval_chain

#=======================================================================================================
# STEP 1: DEFINE THE TEST CASES
#=======================================================================================================


finance_test_cases = [
    {"question": "I'm worried about unexpected job loss. How can PaisaVault help me prepare?", 
     "expected_context": "Emergency fund, Scenario Testing, financial resilience, expenses"},
    {"question": "I received a suspicious email asking for my bank details. What should I do?", 
     "expected_context": "Phishing, fraud, OTP, banking credentials, report"},
    {"question": "My spending always seems to increase with my salary. What can I do to break this cycle?", 
     "expected_context": "Lifestyle inflation, savings, budget, goals, expenses"},
    {"question": "I want to buy a house in 10 years. How can this app assist me in achieving this goal?", 
     "expected_context": "Goals & Savings, Butterfly Effect Simulator, down payment, home loan calculator"},
    {"question": "I'm new to investing and overwhelmed. Where should I begin using PaisaVault?", 
     "expected_context": "Educational resources, FDs, mutual funds, SIP, risk tolerance"},
    {"question": "How does PaisaVault maintain my privacy while analyzing my financial data?", 
     "expected_context": "No PII, no bank details, CSV upload, data control, privacy-centered"},
    {"question": "My credit score is low. Can PaisaVault provide recommendations for improvement?", 
     "expected_context": "CIBIL, credit utilization, payment history, loans, EMI"},
    {"question": "I'm curious about the future value of my investments. How can I simulate this in PaisaVault?", 
     "expected_context": "Compound interest calculator, Butterfly Effect Simulator, LSTM, long-term, projections"},
    {"question": "Can PaisaVault track different types of income besides my salary?", 
     "expected_context": "Rental income, investment interest, gifts, pension, additional income"},
    {"question": "How is PaisaVault different from linking my bank accounts to a budgeting app?", 
     "expected_context": "Manual upload, CSV, privacy, no bank details, control, security"}
]

# Additional test cases 
additional_test_cases = [
    {"question": "I'm new to budgeting. What's a good starting point?", 
     "reference": "Begin by monitoring your income and expenses. Note down all your income sources. Then, categorize your spending into essential (rent, bills) and non-essential (dining, entertainment). Assign budget amounts to each category and regularly track your spending. PaisaVault can simplify this process for you."},
    {"question": "I've heard of the 50/30/20 budget. How does that actually work?", 
     "reference": "The 50/30/20 rule suggests dividing your income into three main categories: 50% for essential needs like rent and groceries, 30% for wants like entertainment and hobbies, and the remaining 20% for savings and paying off debt. It's a good framework for balancing your spending priorities."},
    {"question": "What if I can't save 20% of my salary? What's a realistic savings goal?", 
     "reference": "While 20% is a good target, the important thing is to save consistently, even if it's a smaller amount initially. Start with a percentage you can comfortably manage and gradually increase it as your income rises or expenses decrease. Even 5-10% is a positive step."},
    {"question": "Any tips for keeping track of my spending effectively?", 
     "reference": "Categorize all your expenses, regularly review your transactions, and look for patterns. PaisaVault's automated categorization tools can make this easier by sorting your spending, setting category limits, and alerting you when you approach your budget limits. This helps you understand where your money is going and identify potential areas for savings."},
    {"question": "What are some typical budgeting pitfalls to watch out for?", 
     "reference": "Avoid setting unrealistic budget expectations, remember to include occasional expenses (like insurance premiums), adjust your budget as your situation changes, explore ways to increase income rather than just cutting expenses, and establish an emergency fund for unexpected events. PaisaVault's features can help you steer clear of these common errors."},
    {"question": "I'm trying to cut back on my spending. Where should I start?", 
     "reference": "Begin by separating essential needs from discretionary wants. Review any subscriptions or memberships you rarely use, cut down on dining out, comparison shop for major purchases, and look for energy-saving measures at home. PaisaVault can analyze your spending and offer personalized suggestions for areas to reduce."},
    {"question": "What exactly does 'discretionary spending' mean?", 
     "reference": "Discretionary spending refers to the non-essential expenses you can adjust based on your needs, like entertainment, restaurant meals, hobbies, or subscription services. These are distinct from fixed expenses like rent or utilities. They are typically the easiest expenses to reduce when looking to save money."},
    {"question": "I'm curious about zero-based budgeting. How does that work?", 
     "reference": "In zero-based budgeting, you allocate every rupee of your income to a specific category (expenses, savings, investments) until your budget reaches zero. This approach forces you to be intentional about all your spending and helps eliminate waste. PaisaVault's tools can help you implement this method."},
    {"question": "How much money should I aim to keep in my emergency fund?", 
     "reference": "A good target is 3 to 6 months of essential expenses. If your income is unstable or you have dependents, aim for 6 to 12 months' worth. Don't feel pressured to save this amount immediately. Start with a smaller, manageable goal (like ₹25,000-₹50,000) and gradually build it up."},
    {"question": "What's the best place to keep the money for my emergency fund?", 
     "reference": "Stash your emergency fund in an account that's easily accessible and has minimal risk, like a high-yield savings account or a short-term fixed deposit. Avoid investments with fluctuating values or withdrawal penalties. The goal is to balance accessibility with earning some interest to offset inflation."}
]


for test_case in additional_test_cases:
    '''
    1. LOOPS THROUGH THE ADDITIONAL TEST CASES
    2. EXTRACTS IMPORTANT WORDS ( > 5 CHARS) FROM THE REFERENCE ANSWER
    3. CREATES AN expected_Context FIELD BY JOINING UP TO 5 KEY PHRASES
    4. ADDS THE MODIFIED TEST CASE TO THE MAIN finance_test_cases LIST
    '''
    # Extract key phrases from reference to create expected_context
    words = test_case["reference"].lower().split()
    key_phrases = []
    for i in range(len(words)):
        if len(key_phrases) >= 5:  # Limit to 5 key phrases
            break
        if len(words[i]) > 5 and words[i] not in [phrase.lower() for phrase in key_phrases]:  # Simple heuristic for important words
            key_phrases.append(words[i])
    
    test_case["expected_context"] = ", ".join(key_phrases)
    finance_test_cases.append(test_case)

#=======================================================================================================
# STEP 2: HELPER FUNCTIONS
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

#=======================================================================================================
# STEP 3: RESOURCE MONITORING FUNCTIONS
#=======================================================================================================

# Resource monitoring functions
def start_resource_monitoring():
    """
    1. START TRACKNG MEMORY ALLOCATIONS WITH TRACEMALLOC
    2. RECORD INITIAL CPU TIMES, CPU PERCENTAGE, AND MEMORY USAGE
    3. RETURN A DICTIONARY WITH THE INITIAL METRICS
    4. NOTE THAT THE MEMORY IS CONVERTED TO MB
    """
    tracemalloc.start() # Start memory tracking
    initial_cpu_times = psutil.cpu_times() # Get initial CPU times
    
    return {
        "start_time": time.time(),
        "initial_cpu_times": initial_cpu_times,
        "cpu_percent_start": psutil.cpu_percent(interval=0.1),
        "memory_start": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
    }

def get_resource_usage(monitoring_data):
    """
    1. CALCULATES ELAPSED TIME SINCE MONITORING STARTED
    2. MEASURES CPU TIMES (USER AND SYSTEM) RELATIVE TO THE INITIAL VALUES
    3. GET CURRENT AND PEAK MEMORY USAGE FROM tracemalloc
    4. GET CURRENT PROCESS MEMORY USAGE
    5. RETURN A DICTIONARY WITH ALL THESE METRICS
    """
    current_time = time.time()
    elapsed_time = current_time - monitoring_data["start_time"]
    
    # Get current CPU times
    current_cpu_times = psutil.cpu_times()
    cpu_time_user = current_cpu_times.user - monitoring_data["initial_cpu_times"].user
    cpu_time_system = current_cpu_times.system - monitoring_data["initial_cpu_times"].system
    cpu_time_total = cpu_time_user + cpu_time_system
    
    # Get current memory usage
    current, peak = tracemalloc.get_traced_memory()
    current_memory_mb = current / (1024 * 1024)  # Convert to MB
    peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
    
    # Get process memory info
    process_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
    memory_increase = process_memory - monitoring_data["memory_start"]
    
    return {
        "elapsed_time_seconds": elapsed_time,
        "cpu_time_user_seconds": cpu_time_user,
        "cpu_time_system_seconds": cpu_time_system,
        "cpu_time_total_seconds": cpu_time_total,
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_current_mb": current_memory_mb,
        "memory_peak_mb": peak_memory_mb,
        "process_memory_mb": process_memory,
        "memory_increase_mb": memory_increase
    }


def stop_resource_monitoring():
    """Stop memory tracking"""
    tracemalloc.stop()


#=======================================================================================================
# STEP 4: RAG EVALUATION FUNCTION
#=======================================================================================================


def evaluate_rag_metrics(chain, test_case, evaluator_model):
    """
    Evaluates RAG-specific metrics for a single test case
    1. IMPORT PromptTemplate FROM LANGCHAIN
    2. START RESOURCE MONITORING FOR THIS TEST CASE
    3. MEASURE HOW LONG IT TAKES FOR THE CHATBOT TO RESPOND
    4. EXTRACT THE ANSWER AN CONTEXT FROM THE RESPONSE
    5. RECORD RESOURCE USAGE METRICS
    """
    try:
        from langchain.prompts import PromptTemplate # Import PromptTemplate here to avoid circular imports
        test_monitoring = start_resource_monitoring() # Start resource monitoring for this test case
        
        # Measure response time
        start_time = time.time()
        response = chain.invoke(test_case["question"])
        end_time = time.time()
        response_time = end_time - start_time
        
        result = response['result']  # Extract result
        
        # Get context if available
        if 'source_documents' in response and response['source_documents']:
            context = response['source_documents'][0].page_content
        else:
            context = "Context not available"
        
        # Get resource usage for this test case
        test_resources = get_resource_usage(test_monitoring)
        
        '''
        1. CREATE PROMPT TEMPLATE
        2. CREATING A CHAIN BY PIPING THE PROMPT TO THE EVALUATOR MODEL
        3. INVOKING THE CHAIN WITH HE ACTUAL QUESTION AND ANSWER
        4. EXTRACTING A NUMERICAL SCORE FROM THE MODEL'S RESPONSE
        '''

        #================ METRIC 1: ANSWER RELEVANCY ======================
        relevancy_prompt = PromptTemplate(
            template="""
            You are evaluating a finance chatbot. Rate how relevant the answer is to the question.
            
            Question: {question}
            Answer: {answer}
            
            Rate ONLY with a number from 1-10 (10 being perfectly relevant).
            """,
            input_variables=["question", "answer"]
        )
        
        relevancy_chain = relevancy_prompt | evaluator_model
        relevancy_response = relevancy_chain.invoke({"question": test_case["question"], "answer": result})
        relevancy_score = extract_score(relevancy_response)
        
        #=============== METRIC 2: FAITHFULNESS TO CONTEXT ================
        faithfulness_prompt = PromptTemplate(
            template="""
            You are evaluating a finance chatbot. Rate if the answer is factually consistent with the context.
            
            Context: {context}
            Answer: {answer}
            
            Rate ONLY with a number from 1-10 (10 being completely faithful to the context).
            """,
            input_variables=["context", "answer"]
        )
        
        faithfulness_chain = faithfulness_prompt | evaluator_model
        faithfulness_response = faithfulness_chain.invoke({"context": context, "answer": result})
        faithfulness_score = extract_score(faithfulness_response)
        
        #================ METRIC 3: CONCISENESS ===========================
        conciseness_prompt = PromptTemplate(
            template="""
            You are evaluating a finance chatbot. Rate how concise the answer is while still being complete.
            
            Question: {question}
            Answer: {answer}
            
            Rate ONLY with a number from 1-10 (10 being optimally concise).
            """,
            input_variables=["question", "answer"]
        )
        
        conciseness_chain = conciseness_prompt | evaluator_model
        conciseness_response = conciseness_chain.invoke({"question": test_case["question"], "answer": result})
        conciseness_score = extract_score(conciseness_response)
        
        #================ METRIC 4: ROLE ADHERENCE =======================
        role_prompt = PromptTemplate(
            template="""
            You are evaluating a finance chatbot. Rate how well the answer maintains the role of a finance consultant.
            
            Answer: {answer}
            
            Rate ONLY with a number from 1-10 (10 being perfect adherence to finance consultant role).
            """,
            input_variables=["answer"]
        )
        
        role_chain = role_prompt | evaluator_model
        role_response = role_chain.invoke({"answer": result})
        role_score = extract_score(role_response)
        
        #================= METRIC 5: INFORMATION CORRECTNESS ================
        expected_context = test_case.get("expected_context", "").lower()
        context_lower = context.lower()
        # Check if expected keywords are in the context
        context_match = any(keyword in context_lower for keyword in expected_context.split(", "))
        

        #================= METRIC 6. REFERENCE SIMILARITY EVALUATION ===================
        #(for test cases with reference answers)
        similarity_score = 0
        if "reference" in test_case:
            similarity_prompt = PromptTemplate(
                template="""
                You are evaluating a finance chatbot. Rate how similar the answer is to the reference answer in terms of content and advice.
                
                Question: {question}
                Chatbot Answer: {answer}
                Reference Answer: {reference}
                
                Rate ONLY with a number from 1-10 (10 being perfectly similar in content and advice).
                """,
                input_variables=["question", "answer", "reference"]
            )
            
            similarity_chain = similarity_prompt | evaluator_model
            similarity_response = similarity_chain.invoke({
                "question": test_case["question"], 
                "answer": result, 
                "reference": test_case["reference"]
            })
            similarity_score = extract_score(similarity_response)
        
        return {
            "question": test_case["question"],
            "answer": result,
            "context": context[:200] + "..." if len(context) > 200 else context,  # Truncate long contexts
            "reference": test_case.get("reference", "N/A"),
            "response_time": response_time,
            "relevancy_score": relevancy_score,
            "faithfulness_score": faithfulness_score,
            "conciseness_score": conciseness_score,
            "role_adherence_score": role_score,
            "similarity_score": similarity_score if "reference" in test_case else None,
            "context_match": context_match,
            "resource_usage": test_resources
        }
    except Exception as e:
        print(f"Error evaluating question '{test_case['question']}': {str(e)}")
        return {
            "question": test_case["question"],
            "error": str(e),
            "response_time": 0,
            "relevancy_score": 0,
            "faithfulness_score": 0,
            "conciseness_score": 0,
            "role_adherence_score": 0,
            "similarity_score": 0 if "reference" in test_case else None,
            "context_match": False,
            "resource_usage": {}
        }


#======================================================================================================
# STEP 5: MAIN EVALUATION FUNCTION
#======================================================================================================

def evaluate_chatbot():
    """Main function to evaluate the chatbot on all test cases"""
    print("Setting up evaluation environment...")
    overall_monitoring = start_resource_monitoring() # Start overall resource monitoring
    
    try:
        model = initialize_model(temperature=0.1) # Initialize application model
        evaluator_model = initialize_model(temperature=0.0)  # Initialize evaluator model (separate from application model)
        
        # Use a faster, smaller embedding model for evaluation
        embedding_function = initialize_embedding_function( model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Set up the chain
        print("Loading data and setting up chain...")
        file_path = "generic2.csv"
        docs = docs_preprocessing_helper(file_path)
        db = setup_chroma_db(docs, embedding_function)
        prompt = create_prompt_template()
        chain = create_retrieval_chain(model, db, prompt)
        
        # Evaluate each test case
        results = []
        print("\nEvaluating test cases:")
        for i, test_case in enumerate(finance_test_cases):
            print(f"  [{i+1}/{len(finance_test_cases)}] Testing: {test_case['question']}")
            result = evaluate_rag_metrics(chain, test_case, evaluator_model)
            results.append(result)
            
            # Print resource usage for this test case
            if "resource_usage" in result and result["resource_usage"]:
                ru = result["resource_usage"]
                print(f"    CPU: {ru.get('cpu_percent', 'N/A')}%, Mem: {ru.get('process_memory_mb', 'N/A'):.2f} MB")
        
        overall_resources = get_resource_usage(overall_monitoring) # Get overall resource usage
        stop_resource_monitoring()
        
        valid_results = [r for r in results if "error" not in r] # Calculate average scores
        
        if valid_results:
            avg_metrics = {
                "avg_relevancy": sum(r["relevancy_score"] for r in valid_results) / len(valid_results),
                "avg_faithfulness": sum(r["faithfulness_score"] for r in valid_results) / len(valid_results),
                "avg_conciseness": sum(r["conciseness_score"] for r in valid_results) / len(valid_results),
                "avg_role_adherence": sum(r["role_adherence_score"] for r in valid_results) / len(valid_results),
                "avg_response_time": sum(r["response_time"] for r in valid_results) / len(valid_results),
                "context_match_rate": sum(1 for r in valid_results if r["context_match"]) / len(valid_results) * 100
            }
            
            # Add average similarity score only for test cases with references
            reference_results = [r for r in valid_results if r["similarity_score"] is not None]
            if reference_results:
                avg_metrics["avg_similarity"] = sum(r["similarity_score"] for r in reference_results) / len(reference_results)
        else:
            avg_metrics = {
                "avg_relevancy": 0,
                "avg_faithfulness": 0,
                "avg_conciseness": 0,
                "avg_role_adherence": 0,
                "avg_response_time": 0,
                "context_match_rate": 0,
                "avg_similarity": 0
            }
        
        # Create summary report
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "average_metrics": avg_metrics,
            "overall_resource_usage": overall_resources,
            "detailed_results": results
        }
        
        # Save results
        with open("generic_ollama_evaluation.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create CSV for easier analysis
        df_results = []
        for r in results:
            if "error" not in r:
                row = {
                    "question": r["question"],
                    "relevancy": r["relevancy_score"],
                    "faithfulness": r["faithfulness_score"],
                    "conciseness": r["conciseness_score"],
                    "role_adherence": r["role_adherence_score"],
                    "similarity": r["similarity_score"] if r["similarity_score"] is not None else "N/A",
                    "response_time": r["response_time"],
                    "context_match": r["context_match"],
                    "cpu_percent": r["resource_usage"].get("cpu_percent", "N/A"),
                    "memory_mb": r["resource_usage"].get("process_memory_mb", "N/A")
                }
                df_results.append(row)
        
        pd.DataFrame(df_results).to_csv("generic_chatbot_eval_results.csv", index=False)
        
        # Print summary
        print("\n===== EVALUATION RESULTS =====")
        print(f"Average Relevancy: {avg_metrics['avg_relevancy']:.2f}/10")
        print(f"Average Faithfulness: {avg_metrics['avg_faithfulness']:.2f}/10")
        print(f"Average Conciseness: {avg_metrics['avg_conciseness']:.2f}/10")
        print(f"Average Role Adherence: {avg_metrics['avg_role_adherence']:.2f}/10")
        if "avg_similarity" in avg_metrics:
            print(f"Average Reference Similarity: {avg_metrics['avg_similarity']:.2f}/10")
        print(f"Average Response Time: {avg_metrics['avg_response_time']:.2f} seconds")
        print(f"Context Match Rate: {avg_metrics['context_match_rate']:.1f}%")
        print("==============================")
        
        # Print resource usage summary
        print("\n===== RESOURCE USAGE =====")
        print(f"Total Elapsed Time: {overall_resources['elapsed_time_seconds']:.2f} seconds")
        print(f"CPU Time (User): {overall_resources['cpu_time_user_seconds']:.2f} seconds")
        print(f"CPU Time (System): {overall_resources['cpu_time_system_seconds']:.2f} seconds")
        print(f"Average CPU Usage: {overall_resources['cpu_percent']:.1f}%")
        print(f"Peak Memory Usage: {overall_resources['memory_peak_mb']:.2f} MB")
        print(f"Memory Increase: {overall_resources['memory_increase_mb']:.2f} MB")
        print("==========================\n")
        
        # Print detailed results for any failed tests
        failed_tests = [r for r in results if "error" in r]
        if failed_tests:
            print(f"\n{len(failed_tests)} FAILED TESTS:")
            for i, test in enumerate(failed_tests):
                print(f"  {i+1}. Question: {test['question']}")
                print(f"     Error: {test['error']}")
            print()
        
        return summary
        
    except Exception as e:
        # Make sure to stop resource monitoring even if there's an error
        overall_resources = get_resource_usage(overall_monitoring)
        stop_resource_monitoring()
        
        print(f"Error in evaluation: {str(e)}")
        traceback.print_exc()
        
        return {
            "error": str(e),
            "resource_usage": overall_resources
        }

evaluate_chatbot()