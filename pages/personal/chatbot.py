import streamlit as st
import os
import mysql.connector as sql
import pandas as pd
from transformers import AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

#For loading environment variable from a .env file
load_dotenv() 

#==============================================================================================
#STEP 1: INSTANTIATION
#===============================================================================================
'''
Instantiation model object from ChatGroq Class

-> temperature: controls how random or deterministic the generated o/p is
					lower temp = more predictable, focused and conservative resp
					high temp = more random, diverse and creative responses
'''
model = ChatGroq(
    temperature=0.1,  # Lower temperature for more precise responses
    model_name="mistral-saba-24b",  # Using Mixtral as the Mistral-like model
    # Retrieve API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
)

#=============================================================================================
#STEP 2: EMBEDDING FUNC INITIALIZATION
#=============================================================================================

'''Convert text into vector embeddings for similarity search'''
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

'''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
STEPS 3-8: FOR SQL CHATBOT (DATADIGGER): 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''
#===============================================================================================
#STEP 3: DATABASE CONFIG
#===============================================================================================

# MySQL Configuration for DataDigger
mysql_config = {
    'host': 'localhost',
    'user': 'root',  
    'password': 'themortalinstruments',
    'database': 'expenses_db'
}

#=================================================
#STEP 4: DATABASE INITIALIZATION FUNCTION
#=================================================

'''
1. Connects to MySQL without specifying a database
2. Creates the database if it doesn't exist and selects it
3. Creates the Capstone table with fields for financial transactions if it doesn't exist
4. Checks if the table is empty; if so and if a dummy1.csv file exists:
	a) Loads the CSV data using pandas
	b) Converts dates from DD-MM-YYYY to YYYY-MM-DD format for MySQL
	c) Bulk inserts the data into the Capstone table
5. Closes the connection
'''

def initialize_database():
    mycon = sql.connect( host=mysql_config['host'], user=mysql_config['user'],
                         password=mysql_config['password'] )
    cursor = mycon.cursor()
    
    # Step 1: Create database if it doesn't exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {mysql_config['database']}")
    cursor.execute(f"USE {mysql_config['database']}")
    
    # Step 2: Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Capstone (
        Date DATE,
        Mode VARCHAR(255),
        Category VARCHAR(255),
        Remark VARCHAR(255),
        Amount FLOAT,
        Income_Expense VARCHAR(50),
        Transaction_id VARCHAR(50) PRIMARY KEY
    )
    """)
    
    # Step 3: Check if data needs to be imported by checking count of no.of rows
    cursor.execute("SELECT COUNT(*) FROM Capstone")
    count = cursor.fetchone()[0]
    
    # if no.of rows==0 (ie table empty in mysql) and file exists
    if count == 0 and os.path.exists("data\\dummy1.csv"):
        df = pd.read_csv("data\\dummy1.csv") #Import data from CSV
        
        # Convert DataFrame to list of tuples for bulk insert
        records = []
        for _, row in df.iterrows():
            date_parts = row['Date'].split('-') #Convert date from DD-MM-YYYY to YYYY-MM-DD for MySQL
            mysql_date = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
            
            records.append(( mysql_date, row['Mode'], row['Category'], row['Remark'],
                float(row['Amount']), row['Income_Expense'], row['Transaction_id'] ))
        
        # Bulk insert data
        sql_query = """INSERT INTO Capstone 
                 (Date, Mode, Category, Remark, Amount, Income_Expense, Transaction_id) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        cursor.executemany(sql_query, records)
        mycon.commit()
        print(f"Imported {len(records)} records from data\dummy1.csv")
    
    mycon.close()

#=================================================
#STEP 5: SQL-to-English Conversion
#=================================================
'''
1. Defines a prompt template for converting SQL query results to conversational language
2. Creates a function that:
	a) Takes a question, SQL query, column names, and query results
	b) Formats the prompt with these parameters
	c) Calls the LLM to generate a conversational response
	d) Returns the generated content
'''
sql_to_english_prompt_template = """
You are an expert at converting SQL query results into natural, conversational English responses.

CRITICAL CONVERSION GUIDELINES:
1. Match the exact style of the provided examples
2. Preserve all original data details
3. Use conversational but precise language
4. Follow the specific response patterns demonstrated
5. Always use ₹ instead of $ for displaying amount

Context:
- Original Question: {original_question}
- SQL Query: {sql_query}
- Column Names: {columns}
- Result: {result}

Conversion Rules:
- For single values: Provide context and explain the value
- For single rows: Create a narrative explaining the transaction details
- For multiple rows: Summarize key insights
- For aggregations: Explain the significance of the totals
- Always maintain a helpful, assistant-like tone

Provide ONLY the natural language response. Do not include the original question or query details.
"""

def generate_sql_to_english_response(original_question, sql_query, columns, result):
    # Prepare the prompt
    prompt = PromptTemplate(
        template=sql_to_english_prompt_template, 
        input_variables=[
            "original_question", 
            "sql_query", 
            "columns", 
            "result"
        ]
    )
    
    # Serialize result for passing to LLM (convert to string representation)
    result_str = str(result)
    
    # Create the full input dictionary
    input_dict = {
        "original_question": original_question,
        "sql_query": sql_query,
        "columns": ", ".join(columns),
        "result": result_str
    }
    
    # Use the same Groq model for SQL to English conversion
    response = model.invoke(prompt.format(**input_dict))
    return response.content

#=====================================================
#STEP 6: NATURAL LANGUAGE INTERPRETATION
#=====================================================

'''
1. Handles SQL query results and converts them to natural language
2. Checks if results are empty or contain an error
3. For single rows or single values:
	a) Uses the LLM to generate a response
	b) Falls back to a simple format if that fails
4. For aggregation queries (one column, one row):
	a) Tries to use the LLM
	b) Falls back to a simple format if that fails
5. For multiple rows, returns a brief message (the data will be displayed as a table)
'''
def natural_language_interpretation(original_question, sql_query, columns, result):
    # If result is an error or empty
    if not result or (isinstance(result, list) and isinstance(result[0], str) and result[0].startswith("Error:")):
        return "I couldn't find any information matching your query."
    
    # Convert results to DataFrame for easier processing
    #df = pd.DataFrame(result, columns=columns)
    
    # For single row or single value results
    if len(result) == 1:
        # Use the SQL to English converter
        try:
            return generate_sql_to_english_response( original_question, sql_query, columns, result)
        except Exception as e:
            # Fallback to default interpretation if generation fails
            print(f"SQL to English conversion error: {e}")
            
            row = result[0]
            # If single scalar value
            if len(columns) == 1 and len(row) == 1:
                return f"The result is: {row[0]}"
            
            # Default single row interpretation
            description = "Query Result:\n"
            for col, val in zip(columns, row):
                description += f"- **{col}**: {val}\n"
            return description
    
    # For sum or aggregation queries
    if len(columns) == 1 and len(result[0]) == 1:
        try:
            return generate_sql_to_english_response( original_question, sql_query, columns, result)
        except Exception as e:
            # Fallback if generation fails
            print(f"SQL to English conversion error: {e}")
            total_value = result[0][0]
            return f"Amount corresponding to the query sums to {total_value}"
    
    # If more than 5 rows, don't display detailed text, will use dataframe
    return "Found multiple records. Displaying in table format."

#===============================================================================================
#STEP 7: SQL QUERY EXECTUION AND DATADIGGER PROMPT TEMPLATE
#================================================================================================

def read_sql_query(sql_query, db_config):
    try:
        mycon = sql.connect(**db_config)
        cursor = mycon.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Get column names
        if cursor.description:
            columns = [column[0] for column in cursor.description]
            result = rows
        else:
            columns = []
            result = rows
        
        cursor.close()
        mycon.close()
        return result, columns
    except sql.Error as e:
        return [f"Error: {e}"], []

# Prompt template for DataDigger
data_digger_prompt_template = """
You are an expert in converting English questions to MySQL SQL queries!
The MySQL database has a table named Capstone with the following columns:
- Date (DATE)
- Mode (VARCHAR)
- Category (VARCHAR) - IMPORTANT: This contains specific expense and income categories listed below
- Remark (VARCHAR) - IMPORTANT: This contains specific details about the transaction like 'Hospital', 'Restaurant Name', 'Gas Station', etc.
- Amount (FLOAT)
- Income_Expense (VARCHAR) - Can be 'Income' or 'Expense'
- Transaction_id (VARCHAR) - Primary key

CATEGORY TYPES AND THEIR CLASSIFICATIONS:
# Expense categories:
- Groceries (Expense)
- Education (Expense)
- Healthcare (Expense)
- Transportation (Expense)
- Utilities (Expense)
- Communication (Expense)
- Enrichment (Expense)
- Domestic_Help (Expense)
- Care_Essentials (Expense)
- Financial_Dues (Expense)
- Discretionary (Expense)

# Income categories:
- Salary (Income)
- Pension (Income)
- Rewards (Income)
- Stocks (Income)
- Interest (Income)
- Side_Hustle (Income)
- Gifts (Income)

# Dual-purpose categories (can be either Income or Expense):
- Rent (Dual)
- Miscellaneous (Dual)

IMPORTANT DISTINCTION:
- Category is the specific type of income or expense as listed above
- Remark contains the specific detail or place of transaction (e.g., 'Hospital visit', 'Grocery store', 'College fee')

IMPORTANT ABOUT CASE SENSITIVITY:
- Always use LOWER() function when performing text comparisons to make searches case-insensitive
- Example: LOWER(Remark) LIKE LOWER('%kitchen%') instead of Remark LIKE '%Kitchen%'

IMPORTANT ABOUT SEARCH TERMS:
- When the user asks about transactions related to an item that could appear in either Category OR Remark columns, your query must check BOTH columns using the OR operator
- Example: For "Show me all milk transactions" use: WHERE LOWER(Category) LIKE LOWER('%milk%') OR LOWER(Remark) LIKE LOWER('%milk%')

IMPORTANT ABOUT DATE HANDLING:
- Check if Date column is stored as MySQL native date format (YYYY-MM-DD) or as string in DD-MM-YYYY format
- The Date column is typically stored in MySQL's native YYYY-MM-DD format when imported from CSV, even if the original CSV used DD-MM-YYYY
- For date operations on MySQL native format:
  * Extract year: YEAR(Date) = 2015
  * Extract month: MONTH(Date) = 1 (for January)
  * Extract day: DAY(Date) = 15
  * Current month: MONTH(Date) = MONTH(CURRENT_DATE)
  * Previous month: MONTH(Date) = MONTH(CURRENT_DATE - INTERVAL 1 MONTH)
  * Date range: Date BETWEEN '2015-01-01' AND '2015-12-31'
- If queries return empty results, verify date format with: SELECT Date FROM Transactions LIMIT 5;
- If still uncertain, use pattern matching as fallback:
  * For year 2015: Date LIKE '2015-%'
  * For January: Date LIKE '%-01-%'
  * For 15th day: Date LIKE '%-%-15'

Examples for date handling:
1. For expenses in 2015: WHERE YEAR(Date) = 2015
2. For expenses in January: WHERE MONTH(Date) = 1
3. For expenses on the 15th of any month: WHERE DAY(Date) = 15
4. For expenses in January 2015: WHERE YEAR(Date) = 2015 AND MONTH(Date) = 1
5. For last 30 days: WHERE Date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)

Examples:
1. Question: How many entries of records are present?
   SQL: SELECT COUNT(*) FROM Capstone;

2. Question: Give me the category and amount corresponding to Transaction id TXN129?
   SQL: SELECT Category, Amount FROM Capstone WHERE Transaction_id="TXN129";

3. Question: Display all the rows which contain amount between 100 and 1000 within the first 15 rows.
   SQL: SELECT * FROM Capstone WHERE Amount BETWEEN 100 AND 1000 LIMIT 15;

4. Question: What are the total expenses for each category, and display them in descending order of total expenses?
   SQL: SELECT Category, SUM(Amount) AS TotalExpenses FROM Capstone WHERE Income_Expense = 'Expense' GROUP BY Category ORDER BY TotalExpenses DESC;

5. Question: List all unique categories that have remarks containing the word "auto"
   SQL: SELECT DISTINCT Category FROM Capstone WHERE LOWER(Remark) LIKE LOWER('%auto%');

6. Question: On which date did I make a payment of 3000 to the hospital?
   SQL: SELECT Date FROM Capstone WHERE Amount = 3000 AND LOWER(Remark) LIKE LOWER('%hospital%');

7. Question: What is the total amount spent on healthcare?
   SQL: SELECT SUM(Amount) AS TotalHealthcareExpenses FROM Capstone WHERE LOWER(Category) = LOWER('Healthcare') AND Income_Expense = 'Expense';

8. Question: How much did I spend at restaurants last month?
   SQL: SELECT SUM(Amount) FROM Capstone WHERE LOWER(Remark) LIKE LOWER('%restaurant%') AND Income_Expense = 'Expense' AND MONTH(Date) = MONTH(CURRENT_DATE - INTERVAL 1 MONTH);

9. Question: Show me all my kitchen expenses
   SQL: SELECT * FROM Capstone WHERE LOWER(Remark) LIKE LOWER('%kitchen%');

10. Question: When did I last shop at IKEA?
    SQL: SELECT MAX(Date) as LastVisit FROM Capstone WHERE LOWER(Remark) LIKE LOWER('%ikea%');

11. Question: List all transactions related to education
    SQL: SELECT * FROM Capstone WHERE LOWER(Category) = LOWER('Education') OR LOWER(Remark) LIKE LOWER('%school%') OR LOWER(Remark) LIKE LOWER('%college%') OR LOWER(Remark) LIKE LOWER('%tuition%') OR LOWER(Remark) LIKE LOWER('%course%');

12. Question: Show me all income transactions
    SQL: SELECT * FROM Capstone WHERE Income_Expense = 'Income';

13. Question: What is my total income from side hustles?
    SQL: SELECT SUM(Amount) AS TotalSideHustleIncome FROM Capstone WHERE LOWER(Category) = LOWER('Side_Hustle') AND Income_Expense = 'Income';

14. Question: How much rent did I pay last year?
    SQL: SELECT SUM(Amount) AS TotalRentPaid FROM Capstone WHERE LOWER(Category) = LOWER('Rent') AND Income_Expense = 'Expense' AND YEAR(Date) = YEAR(CURRENT_DATE) - 1;

IMPORTANT NOTES:
- Return ONLY the SQL query without explanations, quotation marks, or text
- Do not include the word 'SQL' or 'sql' in your response
- When filtering by specific places or details, use the Remark column (e.g., LOWER(Remark) LIKE LOWER('%hospital%'))
- When filtering by income/expense types, use the specific Category values listed above (e.g., LOWER(Category) = LOWER('Healthcare'))
- ALWAYS use LOWER() function for text comparisons to ensure case-insensitivity
- When searching for items that could be in either Category or Remark columns, check BOTH columns using OR (e.g., LOWER(Category) LIKE LOWER('%milk%') OR LOWER(Remark) LIKE LOWER('%milk%'))
- For dual-purpose categories (Rent, Miscellaneous), check the Income_Expense column to determine if it's an income or expense

Question: {question}
SQL:
"""

#===============================================================================================
# STEP 8: SQL QUERY GENERATION
#================================================================================================

'''
1. Creates a prompt from the template and the user's question
2. Pipes the prompt to the LLM model using a RunnableSequence
3. Extracts timing information from the response metadata
4. Returns both the generated SQL query and timing details
'''
# Function to get SQL query from Mistral for DataDigger
def get_mistral_response(question, prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    llm_chain = prompt | model  # Using RunnableSequence
    response = llm_chain.invoke({"question": question})
    # Extract timing information from response_metadata
    response_metadata = response.response_metadata
    timing_info = {
        'completion_time': response_metadata.get('token_usage', {}).get('completion_time', 0),
        'prompt_time': response_metadata.get('token_usage', {}).get('prompt_time', 0),
        'queue_time': response_metadata.get('token_usage', {}).get('queue_time', 0),
        'total_time': response_metadata.get('token_usage', {}).get('total_time', 0)
    }
    
    # Return a dictionary with both SQL query and timing information
    return {
        'query': response.content,
        'timing': timing_info
    }

'''
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
STEPS 9-11: FOR DOCUMENT CHATBOT (FINMENTOR): 
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
'''

#==========================================================================================
# STEP 9: DOCUMENT PREPROCESSING for FinMentor
#===========================================================================================

def docs_preprocessing_helper(file):
    loader = CSVLoader(file)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0) #Splits the documents into smaller chunks 
    docs = text_splitter.split_documents(docs)
    return docs

#============================================================================================
# STEP 10: CHROMA DB SETUP 
#=============================================================================================

def setup_chroma_db(docs, embedding_function, persist_directory):
    # Create the persistence directory if it doesn't exist
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        
    # Create a new ChromaDB instance with the specified persistence directory
    db = Chroma.from_documents( docs, embedding_function, persist_directory=persist_directory )
    return db


def create_prompt_template():
    template = """You are a finance consultant chatbot. Answer the customer's questions only using the source data provided.
    Please answer to their specific questions. If you are unsure, say "I don't know, please call our customer support". Keep your answers concise.

    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

#==============================================================================================
# STEP 11: RETRIEVAL CHAIN for FinMentor
#==============================================================================================
def create_retrieval_chain(model, db, prompt):
    '''
    1. Sets up a retrieval QA chain using the "stuff" method
    2. Configures it with the provided LLM model, database, and prompt
    3. Sets the retriever to return only the top result (k=1)
    4. Returns the configured chain
    '''
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff",
                                        retriever=db.as_retriever(search_kwargs={"k": 1}),
                                        chain_type_kwargs=chain_type_kwargs,
    )
    return chain

# Query Chain for FinMentor
def query_chain(chain, query):
    response = chain.invoke(query) #invoke the chain with the query and retrun result field
    return response['result']


def data_digger():

    chat_container = st.container()  #container for chat history (will take most of the screen)
    input_container = st.container() # container at the bottom for the input
    initialize_database() # Initialize the database when the tab is selected
    
    # Set up session state for chat history if not already done
    if "digger_messages" not in st.session_state:
        st.session_state.digger_messages = []
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "query_counter" not in st.session_state:
        st.session_state.query_counter = 0
    
    with input_container:
        prompt = st.chat_input("Ask questions about your transaction history...")
        id = st.session_state.query_counter

        if prompt:
            # Add user message to chat history
            st.session_state.digger_messages.append({"query_id": id, "role": "user", "content": prompt})
            
            with st.spinner("Generating SQL query..."):
                # Get SQL query from Mistral
                result = get_mistral_response(prompt, data_digger_prompt_template)
                sql_query = result['query']
                timing_info = result['timing']
            
            # Execute the SQL query
            try:
                with st.spinner("Executing query..."):
                    result, columns = read_sql_query(sql_query, mysql_config)
                
                # Format the response using natural language interpretation
                if isinstance(result, list) and result and isinstance(result[0], str) and result[0].startswith("Error:"):
                    formatted_response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Error:**\n{result[0]}"
                    st.session_state.digger_messages.append({"query_id": id, "role": "assistant", "content": formatted_response})
                    st.session_state.query_counter += 1 # Increment the counter for the next query
                else:
                    # Call natural language interpretation
                    natural_language_response = natural_language_interpretation(prompt, sql_query, columns, result)
                    
                    # Format the response with SQL query and natural language explanation
                    formatted_response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Insights:**\n{natural_language_response}"
                    
                    # Add assistant response to chat history
                    st.session_state.digger_messages.append({"query_id": id, "role": "assistant", "content": formatted_response})
                    st.session_state.query_counter += 1 # Increment the counter for the next query
    
                    # If result has rows, create and display a DataFrame
                    if result and len(result) > 0:
                        df = pd.DataFrame(result, columns=columns) # Create DataFrame
                        st.session_state.dataframes[id] = df # Store the dataframe in session state, associated with the current query_id
            
            except Exception as e:
                formatted_response = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n**Error:**\n{str(e)}"
                st.session_state.digger_messages.append({"query_id": id, "role": "assistant", "content": formatted_response})
    
    # Use the chat container to display all messages
    with chat_container:
        for i, message in enumerate(st.session_state.digger_messages):

            check = message["query_id"]
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if (message["role"] == "assistant" and "query_id" in message
                    and message["query_id"] in st.session_state.dataframes ):
                    st.dataframe(st.session_state.dataframes[check])


def fin_mentor():
    chat_container = st.container() #container for chat history 
    input_container = st.container() #container for input
    
    # Set up session state for chat history if not already done
    if "mentor_messages" not in st.session_state:
        st.session_state.mentor_messages = []
    
    # Only load and process documents once for FinMentor with a separate persistence directory
    if "mentor_chain" not in st.session_state:
        with st.spinner("Loading model and data (this will only happen once)..."):
            file_path = "data\\custom.csv"
            docs = docs_preprocessing_helper(file_path)
            
            # Use a separate persistence directory for FinMentor
            mentor_persist_dir = "./chroma_fin_mentor"
            db = setup_chroma_db(docs, embedding_function, mentor_persist_dir)
            
            prompt = create_prompt_template()
            st.session_state.mentor_chain = create_retrieval_chain(model, db, prompt)
    
    with input_container:
        prompt = st.chat_input("What is your finance question?")
        
        if prompt:
            # Add user message to chat history
            st.session_state.mentor_messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response = query_chain(st.session_state.mentor_chain, prompt)
                
            # Add assistant response to chat history
            st.session_state.mentor_messages.append({"role": "assistant", "content": response})
    
    with chat_container:
        for message in st.session_state.mentor_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# Main function for the custom bot with tabs
def custom_bot():
    """Custom Bot page with top navigation for two different bots"""
    st.markdown(
    """<style>
        * {
            font-family: Verdana, sans-serif !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            justify-content: center;
            gap: 8px;
            border-bottom: 2px solid #4e807c;
        }
        .stTabs [data-baseweb="tab"] div {
            height: 50px;
            width: 380px;
            background-color: #78c4be; 
            color: #333333;          
            border-radius: 8px 8px 0 0;
            padding: 10px;
            text-align: center;
            font-weight: bold !important;
            font-size: 18px !important;
            cursor: pointer;
        }
        .stTabs [aria-selected="true"] div {
            background-color:  #4e807c; 
            color: white;           
            font-weight: bold !important;
        }
        /* Make the chat input sticky to the bottom */
        .stChatInputContainer {
            position: sticky;
            bottom: 0;
            background-color: white;
            z-index: 1;
            padding-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    st.write("## 💬 Financial Assistant") 
    st.write("Choose between two specialized bots: one that analyzes your transaction history for insights, and another that provides expert financial advice.")
    
    # Create tabs for bot versions in the top navigation
    tab1, tab2 = st.tabs(["DataDigger📈: History-Based", "FinMentor🧠: Intelligence-Based"])
    
    # Content for Bot Version 1
    with tab1:
        st.write("#### Welcome to DataDigger!")
        st.write("Your financial history, decoded with precision.")
        data_digger()
    
    # Content for Bot Version 2
    with tab2:
        st.write("#### Welcome to FinMentor!")
        st.write("Smart financial advice, simplified.")
        fin_mentor()