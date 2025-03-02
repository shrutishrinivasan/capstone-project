import streamlit as st
from utils.auth import register_user, login_user

def login_page():
    """Login section of the landing page"""
    st.title("Account Access")
    st.markdown(
    """
        <style>
            label {
                font-size: 18px !important;
                font-weight: bold !important;
            }
            .stTextInput>div>div>input, .stButton>button {
                width: 100%;
            }
            .stTabs [data-baseweb="tab-list"] {
                justify-content: center;
            }
            .stTabs [data-baseweb="tab"] div {
                height: 50px;
                width: 200px;
                white-space: pre-wrap;
                background-color: #444444;
                border-radius: 4px 4px 0 0;
                gap: 1px;
                padding: 10px;
                font-size: 20px !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        st.write("### Login to Your Account")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if username and password:
                success, message = login_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("Please enter both username and password.")
    
    with tab2:
        st.subheader("Create a New Account")
        new_username = st.text_input("Choose a Username", key="signup_username")
        new_password = st.text_input("Choose a Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up"):
            if new_username and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                        st.info("You can now log in with your new account.")
                    else:
                        st.error(message)
            else:
                st.warning("Please fill in all fields.")    