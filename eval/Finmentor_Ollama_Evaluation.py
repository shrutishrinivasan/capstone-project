#Finmentor_Ollama_Evaluation.py
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

# Finance-specific test cases from paste-2.txt
finance_test_cases = [
    {
        "question": "PaisaVault emphasizes the 'butterfly effect' in finance. Could you elaborate on what this means in the context of the app?",
        "reference": "The butterfly effect refers to the idea that small financial choices today, like skipping a daily latte, can have a significant impact on your long-term financial well-being, similar to how a butterfly flapping its wings can supposedly cause a hurricane. PaisaVault uses this concept to show how small daily decisions compound over time.",
        "expected_context": "butterfly effect, financial choices, long-term impact, compounding, decisions"
    },
    {
        "question": "How does PaisaVault utilize AI to help users make better financial decisions?",
        "reference": "PaisaVault incorporates LSTM, a type of artificial intelligence, to analyze past financial data, predict future outcomes, and provide personalized recommendations for improvement. This technology powers features like the Butterfly Effect Simulator and the personalized AI assistant.",
        "expected_context": "LSTM, AI, financial data, predictions, personalized recommendations"
    },
    {
        "question": "What's the primary difference between PaisaVault and other financial apps in terms of data privacy?",
        "reference": "Unlike many finance apps that require linking bank accounts or sharing personal information, PaisaVault prioritizes user privacy. It doesn't collect personally identifiable information (PII) and allows users to manually upload their data, giving them full control.",
        "expected_context": "privacy, PII, manual upload, data control, no bank linking"
    },
    {
        "question": "If I'm a complete beginner with investing, how can PaisaVault assist me?",
        "reference": "PaisaVault offers educational resources and goal-setting tools to help beginners understand basic investment principles. It also simulates how different saving and investment scenarios could affect their long-term financial health. However, it doesn't give specific investment advice.",
        "expected_context": "educational resources, beginners, goal-setting, investment principles, simulations"
    },
    {
        "question": "Explain how the Financial Scenario Tester in PaisaVault can help someone prepare for unforeseen circumstances.",
        "reference": "The Financial Scenario Tester allows users to simulate different financial situations, like a job loss or medical emergency, to assess their financial resilience. This helps them identify potential weaknesses in their financial plan and take proactive steps to strengthen their safety net.",
        "expected_context": "Financial Scenario Tester, resilience, simulations, emergencies, financial plan"
    },
    {
        "question": "How can PaisaVault help someone create and stick to a budget?",
        "reference": "It's recommended to have 3-6 months of essential expenses saved in an easily accessible emergency fund.",
        "expected_context": "budget, expenses, tracking, recommendations, financial habits"
    },
    {
        "question": "What are the key metrics displayed on PaisaVault's dashboard?",
        "reference": "The dashboard visualizes key information like income, expenses, net savings, saving rate, progress towards goals, and financial health scores. It provides a comprehensive overview of your financial situation.",
        "expected_context": "dashboard, metrics, income, expenses, financial health scores"
    },
    {
        "question": "Besides tracking expenses, what other ways does PaisaVault help users improve their financial health?",
        "reference": "Beyond expense tracking, PaisaVault provides personalized advice, simulates the long-term effects of financial decisions, compares spending habits to benchmarks, offers educational resources, and tests financial resilience, all contributing to improved financial habits.",
        "expected_context": "personalized advice, simulations, benchmarks, educational resources, resilience"
    },
    {
        "question": "I'm interested in using PaisaVault but not ready to create an account. What resources are available to me?",
        "reference": "Even without an account, you can access financial calculators, read educational articles, see demos of the app's features, and interact with a generic financial chatbot on PaisaVault's landing page.",
        "expected_context": "no account, calculators, articles, demos, generic chatbot"
    },
    {
        "question": "What is the difference between the generic financial bot and the personalized AI assistant within PaisaVault?",
        "reference": "The generic bot available on the landing page answers general financial questions. The personalized assistant, accessible after creating an account, is trained on your individual financial data to provide tailored insights and recommendations.",
        "expected_context": "generic bot, personalized assistant, account, tailored insights, financial data"
    }
]

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
        from langchain.prompts import PromptTemplate  # Import PromptTemplate here to avoid circular imports
        
        # Start CPU and memory tracking
        process = psutil.Process(os.getpid())
        tracemalloc.start()
        cpu_percent_start = process.cpu_percent()
        memory_start = tracemalloc.get_traced_memory()[0]
        
        # Measure response time
        start_time = time.time()
        response = chain.invoke(test_case["question"])
        end_time = time.time()
        response_time = end_time - start_time
        
        # Get CPU and memory usage after response
        cpu_percent_end = process.cpu_percent()
        memory_current, memory_peak = tracemalloc.get_traced_memory()
        memory_used = memory_current - memory_start
        tracemalloc.stop()  # Stop tracemalloc for this iteration
        
        # Extract result
        result = response['result']
        
        # Get context if available
        if 'source_documents' in response and response['source_documents']:
            context = response['source_documents'][0].page_content
        else:
            context = "Context not available"
        
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
        context_match = any(keyword.strip() in context_lower for keyword in expected_context.split(","))
        
        #================= METRIC 6. REFERENCE SIMILARITY EVALUATION ===================
        #(for test cases with reference answers)
        reference = test_case.get("reference", "")
        reference_similarity_prompt = PromptTemplate(
            template="""
            You are evaluating a finance chatbot. Rate how similar the answer is to the reference answer in terms of content and meaning.
            
            Reference Answer: {reference}
            Chatbot Answer: {answer}
            
            Rate ONLY with a number from 1-10 (10 being very similar to the reference).
            """,
            input_variables=["reference", "answer"]
        )
        
        reference_chain = reference_similarity_prompt | evaluator_model
        reference_response = reference_chain.invoke({"reference": reference, "answer": result})
        reference_score = extract_score(reference_response)
        
        return {
            "question": test_case["question"],
            "answer": result,
            "context": context[:200] + "..." if len(context) > 200 else context,  # Truncate long contexts
            "reference": reference,
            "response_time": response_time,
            "relevancy_score": relevancy_score,
            "faithfulness_score": faithfulness_score,
            "conciseness_score": conciseness_score,
            "role_adherence_score": role_score,
            "reference_similarity_score": reference_score,
            "context_match": context_match,
            "cpu_percent": cpu_percent_end - cpu_percent_start,
            "memory_used_bytes": memory_used,
            "memory_used_mb": memory_used / (1024 * 1024)  # Convert to MB for readability
        }
    except Exception as e:
        print(f"Error evaluating question '{test_case['question']}': {str(e)}")
        tracemalloc.stop()  # Ensure tracemalloc is stopped in case of error
        return {
            "question": test_case["question"],
            "error": str(e),
            "response_time": 0,
            "relevancy_score": 0,
            "faithfulness_score": 0,
            "conciseness_score": 0,
            "role_adherence_score": 0,
            "reference_similarity_score": 0,
            "context_match": False,
            "cpu_percent": 0,
            "memory_used_bytes": 0,
            "memory_used_mb": 0
        }



#======================================================================================================
# STEP 5: MAIN EVALUATION FUNCTION
#======================================================================================================

def evaluate_chatbot():
    """Main function to evaluate the chatbot on all test cases"""
    print("Setting up evaluation environment...")
    
    try:
        # Start overall resource monitoring
        overall_start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_cpu = process.cpu_percent()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Initialize application model
        model = initialize_model(temperature=0.1)
        
        # Initialize evaluator model (separate from application model)
        evaluator_model = initialize_model(temperature=0.0)
        
        # Use a faster, smaller embedding model for evaluation
        embedding_function = initialize_embedding_function(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Set up the chain
        print("Loading data and setting up chain...")
        file_path = "C:\\Users\\Rhea Pandita\\Downloads\\capstone-project-main\\data\\custom2.csv"
        docs = docs_preprocessing_helper(file_path)
        db = setup_chroma_db(docs, embedding_function)
        prompt = create_prompt_template()
        chain = create_retrieval_chain(model, db, prompt)
        
        # Evaluate each test case
        results = []
        total_cpu = 0
        total_memory = 0
        
        print("\nEvaluating test cases:")
        for i, test_case in enumerate(finance_test_cases):
            print(f"  [{i+1}/{len(finance_test_cases)}] Testing: {test_case['question']}")
            result = evaluate_rag_metrics(chain, test_case, evaluator_model)
            results.append(result)
            total_cpu += result["cpu_percent"]
            total_memory += result["memory_used_mb"]
        
        # Calculate average scores
        valid_results = [r for r in results if "error" not in r]
        
        if valid_results:
            avg_metrics = {
                "avg_relevancy": sum(r["relevancy_score"] for r in valid_results) / len(valid_results),
                "avg_faithfulness": sum(r["faithfulness_score"] for r in valid_results) / len(valid_results),
                "avg_conciseness": sum(r["conciseness_score"] for r in valid_results) / len(valid_results),
                "avg_role_adherence": sum(r["role_adherence_score"] for r in valid_results) / len(valid_results),
                "avg_reference_similarity": sum(r["reference_similarity_score"] for r in valid_results) / len(valid_results),
                "avg_response_time": sum(r["response_time"] for r in valid_results) / len(valid_results),
                "context_match_rate": sum(1 for r in valid_results if r["context_match"]) / len(valid_results) * 100,
                "avg_cpu_percent": total_cpu / len(valid_results),
                "avg_memory_used_mb": total_memory / len(valid_results)
            }
        else:
            avg_metrics = {
                "avg_relevancy": 0,
                "avg_faithfulness": 0,
                "avg_conciseness": 0,
                "avg_role_adherence": 0,
                "avg_reference_similarity": 0,
                "avg_response_time": 0,
                "context_match_rate": 0,
                "avg_cpu_percent": 0,
                "avg_memory_used_mb": 0
            }
        
        # Get overall resource usage
        overall_end_time = time.time()
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        final_cpu = process.cpu_percent()
        
        overall_metrics = {
            "total_execution_time": overall_end_time - overall_start_time,
            "total_memory_increase_mb": final_memory - initial_memory,
            "avg_cpu_overall": (final_cpu - initial_cpu) / 2  # Approximation of average
        }
        
        # Create summary report
        summary = {
            "average_metrics": avg_metrics,
            "overall_resource_usage": overall_metrics,
            "detailed_results": results
        }
        
        # Save results
        with open("finmentor_ollama_evaluation.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n===== EVALUATION RESULTS =====")
        print(f"Average Relevancy: {avg_metrics['avg_relevancy']:.2f}/10")
        print(f"Average Faithfulness: {avg_metrics['avg_faithfulness']:.2f}/10")
        print(f"Average Conciseness: {avg_metrics['avg_conciseness']:.2f}/10")
        print(f"Average Role Adherence: {avg_metrics['avg_role_adherence']:.2f}/10")
        print(f"Average Reference Similarity: {avg_metrics['avg_reference_similarity']:.2f}/10")
        print(f"Average Response Time: {avg_metrics['avg_response_time']:.2f} seconds")
        print(f"Context Match Rate: {avg_metrics['context_match_rate']:.1f}%")
        print("\n===== RESOURCE USAGE =====")
        print(f"Average CPU Usage: {avg_metrics['avg_cpu_percent']:.2f}%")
        print(f"Average Memory Usage: {avg_metrics['avg_memory_used_mb']:.2f} MB per query")
        print(f"Total Execution Time: {overall_metrics['total_execution_time']:.2f} seconds")
        print(f"Total Memory Increase: {overall_metrics['total_memory_increase_mb']:.2f} MB")
        print("==============================\n")
        
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
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    evaluate_chatbot()