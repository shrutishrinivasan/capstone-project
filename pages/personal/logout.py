import streamlit as st
from utils.auth import logout_user
    
def logout():
    """Logout page"""
    st.title("Log Out")
    st.write("Are you sure you want to log out?")
    
    if st.button("Yes, Log Out"):
        logout_user()
        st.success("You have been logged out successfully.")
        st.rerun()