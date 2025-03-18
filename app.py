import streamlit as st
from utils.nav import horizontal_navbar, vertical_navbar, sub_menu

# pre-login
from pages.generic.about import about_page
from pages.generic.features import features_page
from pages.generic.tools import tools_page
from pages.generic.learn import learn_page
from pages.generic.bot import bot_page
from pages.generic.login import login_page

# post-login
from pages.personal.start import getting_started
from pages.personal.upload import upload_data
from pages.personal.overview import overview
from pages.personal.expense import expense
from pages.personal.saving import goals_savings
from pages.personal.chatbot import custom_bot
from pages.personal.analysis import performance_analysis
from pages.personal.finance import financial_suggestions
from pages.personal.edu import education
from pages.personal.logout import logout
from pages.personal.butterfly import butterfly
from pages.personal.scenario import scenario

# Set page configuration
st.set_page_config(
    page_title="PaisaVault",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Track navigation state
if "menu_state" not in st.session_state:
    st.session_state.menu_state = "main"

# Apply custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E24;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3A3B46;
    };
</style>
""", unsafe_allow_html=True)

# Only show sidebar if logged in
if st.session_state.logged_in:
    st.sidebar.markdown("", unsafe_allow_html=True)

# Main app logic
def main():
    # Check if user is logged in
    if not st.session_state.logged_in:
        # Display landing page with horizontal navbar
        selected_tab = horizontal_navbar()
        
        # Display content based on selected tab
        if selected_tab == "About":
            about_page()
        elif selected_tab == "Features":
            features_page()
        elif selected_tab == "Tools":
            tools_page()
        elif selected_tab == "Bot":
            bot_page()
        elif selected_tab == "Learn":
            learn_page()
        elif selected_tab == "Log In":
            login_page()
    else:
        if st.session_state.menu_state == "main":
            # Display dashboard with vertical navbar
            selected_tab = vertical_navbar()
            
            # Display content based on selected tab
            if selected_tab == "Getting Started":
                getting_started()
            elif selected_tab == "Upload Data":
                upload_data()
            elif selected_tab == "Overview":
                overview()
            elif selected_tab == "Income/Expense":
                expense()
            elif selected_tab == "Goals & Savings":
                goals_savings()
            elif selected_tab == "Custom Bot":
                custom_bot()
            elif selected_tab == "Performance Analysis":
                performance_analysis()
            elif selected_tab == "Financial Suggestions":
                financial_suggestions()
            elif selected_tab == "Education":
                education()
            elif selected_tab == "Log Out":
                logout()
        else:
            selected_tab = sub_menu()

            if selected_tab == "Butterfly Effect":
                butterfly()
            elif selected_tab == "Scenario Testing":
                scenario()
    

if __name__ == "__main__":
    main()