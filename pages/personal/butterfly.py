import streamlit as st

def butterfly():
    """Butterfly Effect page"""
    st.title("Butterfly Effect Simulator")

    if st.button("⬅ Back to Dashboard"):
                st.session_state.menu_state = "main"
                st.rerun()