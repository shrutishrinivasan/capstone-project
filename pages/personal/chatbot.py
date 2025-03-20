import streamlit as st

def data_digger():
    """History Based Bot"""
    st.write("Your financial history, decoded with precision.")

def fin_mentor():
    """Intelligence-Based Bot"""
    st.write("Smart financial advice, simplified.")
    
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
    </style>
    """, unsafe_allow_html=True)

    st.write("## 💬 Financial Assistant") 
    st.write("Choose between two specialized bots: one that analyzes your transaction history for insights, and another that provides expert financial advice.")
    
    # Create tabs for bot versions in the top navigation
    tab1, tab2 = st.tabs(["DataDigger🔍: History-Based", "FinMentor🧠: Intelligence-Based"])
    
    # Content for Bot Version 1
    with tab1:
        st.write("#### Welcome to DataDigger!")
        data_digger()
    
    # Content for Bot Version 2
    with tab2:
        st.write("#### Welcome to FinMentor!")
        fin_mentor()
