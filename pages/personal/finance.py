import streamlit as st

def financial_suggestions():
    """Financial Suggestions page"""
    st.session_state.menu_state = "finance_menu"
    st.rerun()