import streamlit as st
import pandas as pd
import datetime
import re
import base64 

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def expense():
    """Expense tracking page with category buttons and transaction entry popup"""
    
    st.markdown("""
    <style>
        * {
        font-family: Verdana, sans-serif !important;
        }
        div.stButton > button {
            background-color: #ffd166; 
            color: black; 
            font-size: 16px;
            width: 150px; 
            height: 20px; 
            margin-top: 5px;
            border-radius: 10px; 
            border: none; 
        }
        div.stButton > button:hover {
            background-color: #8c5818;
            color: white;
        }
        div.stButton > button:focus {
            color: white !important;
            background-color: #8c5818;
        }
        div[data-testid="stForm"] {
            border: 2px solid white;  
            border-radius: 15px;        
            background-color: #191a1c;  
        }
        div[data-testid="stForm"] > div > div > button[kind="primary"] {
            background-color: #834da3;    /* Purple background */
            color: #ffffff;               /* White text */
            border: none;                 /* No border */
            border-radius: 8px;           /* Rounded corners */
            padding: 8px 20px;            /* Button padding */
            font-weight: bold;            /* Bold text */
            width: 100%;                  /* Full width */
            cursor: pointer;              /* Pointer cursor */
            transition: background-color 0.3s ease;
        }
        div[data-testid="stForm"] > div > div > button[kind="primary"]:hover {
            background-color: #5e358a;    /* Darker purple on hover */
        }
        div[data-testid="stForm"] > div > div > button[kind="primary"]:focus {
            background-color: #5e358a;    /* Same color for click */
            outline: none;
        }        
    </style>
    """, unsafe_allow_html=True)
    
    st.write("## 💳 Record Your Income/Expense")
    
    # Initialize session state for finance dataframe if not exists
    if 'finance_df' not in st.session_state:
        st.session_state.finance_df = pd.DataFrame(columns=[
            'Date', 'Mode', 'Category', 'Remark', 'Amount', 'Income_Expense', 'Transaction_id'
        ])
    
    # Category descriptions for information
    category_descriptions = {
        "Rent": "Housing expenses including rent, mortgage payments, property taxes, and maintenance costs. Also income in the form of rent from tenants.",
        "Groceries": "Food and household items purchased for daily consumption.",
        "Healthcare": "Medical expenses including insurance, doctor visits, medications, and procedures.",
        "Transportation": "Expenses related to commuting, vehicle maintenance, fuel, and public transportation.",
        "Utilities": "Household utilities including electricity, water, gas, heating, and cooling.",
        "Communication": "Phone, internet, cable, and other communication-related expenses.",
        "Education": "Educational expenses including tuition, books, courses, and learning materials.",
        "Enrichment": "Expenses for activities that enhance skills, fitness, or personal growth, including dance, music, art classes, sports coaching, entrance exam preparation, gym memberships, yoga, etc.",
        "Domestic_Help": "Payments for household staff including maids, cooks, chauffeurs, gardeners, and nannies.",
        "Care_Essentials": "Expenses on personal care, hygiene products, nursery needs, makeup, pet care, and related essentials.",
        "Financial_Dues": "Debt payments, loan installments, credit card payments, emergency fund, retirement savingsand financial fees.",
        "Discretionary": "Non-essential expenses like entertainment, dining out, hobbies, and leisure activities.",
        "Salary": "Primary income earned through employment, including wages, bonuses, and incentives.",
        "Side_Hustle": "Supplementary income from freelancing, consulting, small businesses, or part-time work.",
        "Pension": "Retirement income from pension plans, provident funds, or other post-retirement benefits.",
        "Interest": "Earnings from savings accounts, fixed deposits, recurring deposits, and similar investments.",
        "Rewards": "Cashback, loyalty points, vouchers, and other reward-based earnings from purchases or promotions.",
        "Stocks": "Income from share trading, mutual funds, stock dividends, bonds, or equity investments.",
        "Gifts": "Monetary gifts received during family events like weddings, housewarming, or festive occasions.",
        "Miscellaneous": "Expenses and sources of income that don't fit into other categories.",
    }

    category_images = {
        "Rent": get_base64_image("static/icons/rent.png"),
        "Groceries":  get_base64_image("static/icons/grocery.png"),
        "Education":  get_base64_image("static/icons/edu.png"),
        "Healthcare":  get_base64_image("static/icons/health.png"),
        "Transportation":  get_base64_image("static/icons/transport.png"),
        "Utilities":  get_base64_image("static/icons/utility.png"),
        "Communication":  get_base64_image("static/icons/comm.png"),
        "Financial_Dues":  get_base64_image("static/icons/obligation.png"),
        "Discretionary":  get_base64_image("static/icons/discretionary.png"),
        "Domestic_Help":  get_base64_image("static/icons/maid.png"),
        "Care_Essentials":  get_base64_image("static/icons/care.png"),
        "Salary":  get_base64_image("static/icons/salary.png"),
        "Rewards":  get_base64_image("static/icons/reward.png"),
        "Pension":  get_base64_image("static/icons/pension.png"),
        "Interest":  get_base64_image("static/icons/interest.png"),
        "Stocks":  get_base64_image("static/icons/stock.png"),
        "Side_Hustle":  get_base64_image("static/icons/side.png"),
        "Enrichment":  get_base64_image("static/icons/skill.png"),
        "Gifts":  get_base64_image("static/icons/gift.png"),
        "Miscellaneous":  get_base64_image("static/icons/misc.png")
    }

    category_types = {
        # Expense categories 
        "Groceries": "expense",
        "Education": "expense",
        "Healthcare": "expense",
        "Transportation": "expense",
        "Utilities": "expense",
        "Communication": "expense",
        "Enrichment": "expense",
        "Domestic_Help": "expense",
        "Care_Essentials": "expense",
        "Financial_Dues": "expense",
        "Discretionary": "expense",
        
        # Income categories 
        "Salary": "income",
        "Pension": "income",
        "Rewards": "income",
        "Stocks": "income",
        "Interest": "income",
        "Side_Hustle": "income",
        "Gifts": "income",
        
        # Dual-purpose categories 
        "Rent": "dual",
        "Miscellaneous": "dual",
    }

    # Function to generate next transaction ID
    def generate_transaction_id():
        if st.session_state.finance_df.empty:
            return "TXN1"
        else:
            # Extract existing transaction IDs
            existing_ids = st.session_state.finance_df['Transaction_id'].tolist()
            txn_numbers = []
            
            # Extract numeric parts from IDs
            for txn_id in existing_ids:
                if isinstance(txn_id, str) and txn_id.startswith("TXN"):
                    match = re.search(r'TXN(\d+)', txn_id)
                    if match:
                        txn_numbers.append(int(match.group(1)))
            
            # Find the maximum number and increment by 1
            if txn_numbers:
                next_number = max(txn_numbers) + 1
            else:
                next_number = 1
                
            return f"TXN{next_number}"
    
    # Function to add transaction
    def add_transaction(category):
        # Create a form for the transaction details
        with st.form(key=f"transaction_form_{category}"):
            st.subheader(f"{category} Transaction")
            st.info(category_descriptions[category])
            
            # Form fields
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Date", value=datetime.date.today())
                mode = st.selectbox("Payment Mode", options=[
                    "Cash", "Credit Card", "Debit Card", "UPI", "Bank Transfer", 
                    "Cheque", "Mobile Wallet", "Automatic Payment", "Net Banking", "Other"
                ])
                transaction_type = st.radio("Type", options=["Expense", "Income"], horizontal=True)
            
            with col2:
                amount = st.number_input("Amount", min_value=0.01, format="%.2f", step=10.0)
                remark = st.text_area("Remark", placeholder="Add details about this transaction...", height=100)
            
            # Generate transaction ID automatically
            transaction_id = generate_transaction_id()
            
            # Submit button
            submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
            with submit_col2:
                submitted = st.form_submit_button("Save Transaction", use_container_width=True)
            
            if submitted:
                # Create new transaction
                new_transaction = pd.DataFrame({
                    'Date': [date.strftime('%Y-%m-%d')],
                    'Mode': [mode],
                    'Category': [category],
                    'Remark': [remark],
                    'Amount': [amount], # if transaction_type == "Income" else -amount],  # Negative for expenses
                    'Income_Expense': [transaction_type],
                    'Transaction_id': [transaction_id]
                })
                
                # Add to dataframe
                st.session_state.finance_df = pd.concat([st.session_state.finance_df, new_transaction], ignore_index=True)
                
                # Success message
                st.success(f"Transaction {transaction_id} recorded successfully!")
                st.info("Please check upload data section to see the manually entered data")
                st.rerun()
    
    st.markdown("""
        ##### Select a category to record your transaction
        - Categories with a <span style="color: white;">**white**</span> background serve as both sources of **income** and **expense**.  
        - Categories with a <span style="color: #e1b2ed;">**purple**</span> background are primarily for **expenses**.  
        - Categories with a <span style="color: #96d99b">**green**</span> background are primarily for **income**.  
    """, unsafe_allow_html=True)


    # Create rows of buttons 
    categories = list(category_descriptions.keys())

    # Button Generation Code
    for i in range(0, len(categories), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(categories):
                category = categories[i + j]
                cat_type = category_types[category]
                indicator_color = {
                        "expense": "#e1b2ed",  # red
                        "income": "#96d99b",   # green
                        "dual": "white"      # yellow
                    }[cat_type]
                
                with cols[j]:
                    # Image Container
                    st.markdown(f"""
                    <div style="
                        background-color: {indicator_color};
                        border-radius: 10px;
                        padding: 5px;
                        text-align: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                        width: 150px;
                        margin-bottom: 1px;">
                        <img src="data:image/png;base64,{category_images[category]}" 
                            width="100" height="100" style="margin-bottom: 5px;">
                    </div>
                    """, unsafe_allow_html=True)

                    if '_' in category:
                        temp = category.replace('_',' ')
                    else:
                        temp = category

                    if st.button(f"{temp}"):
                        st.session_state.selected_category = category

    # Display transaction form for selected category
    if 'selected_category' in st.session_state:
        add_transaction(st.session_state.selected_category)
        
    # Recent transactions display
    if not st.session_state.finance_df.empty:
        st.write("### Recent Transactions")
        
        # # Sort by date (most recent first) and display last 5
        # recent_df = st.session_state.finance_df.sort_values(by='Date', ascending=False).head(5)

        st.session_state.finance_df['Date'] = pd.to_datetime(st.session_state.finance_df['Date'], errors='coerce')
        recent_df = st.session_state.finance_df.sort_values(by='Date', ascending=False).head(5)
        
        # Display with custom formatting
        for idx, row in recent_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.write(f"**{row['Date']}**")
                    st.write(f"{row['Category']}")
                with col2:
                    st.write(f"**{row['Remark']}**")
                    st.write(f"Payment: {row['Mode']}")
                with col3:
                    amount = float(row['Amount'])
                    color = "green" if row['Income_Expense']=='Income' else "red"
                    sign = "+" if row['Income_Expense']=='Income' else "-"
                    st.markdown(f"<h3 style='color: {color}; text-align: right'>{sign}{amount:.2f}</h3>", unsafe_allow_html=True)
                    st.write(f"<p style='text-align: right'>{row['Transaction_id']}</p>", unsafe_allow_html=True)
                st.divider()