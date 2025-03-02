import streamlit as st

def scenario():
    """Scenario Testing page"""
    st.title("Financial Scenario Tester")

    if st.button("⬅ Back to Dashboard"):
                st.session_state.menu_state = "main"
                st.rerun()