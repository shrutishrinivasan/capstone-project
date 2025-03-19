import streamlit as st
import json

def load_faq_data():
    with open(r'C:\Users\USER\Desktop\capstone_project\data\finance.json', 'r') as file:
        return json.load(file)

def get_response(user_question, faq_data):
    user_keywords = user_question.lower().split()
    for question, answer in faq_data.items():
        question_keywords = question.lower().split()
        if any(word in question_keywords for word in user_keywords):
            return answer
    return "Sorry, I couldn't find an answer to your query."

def bot_page():
    """Bot section of the landing page"""

    # Set the font for the entire page
    st.markdown("""
    <style>
    * {
        font-family: Verdana, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Section header with inline horizontal line
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: -20px; margin-top: -10px">
      <h3 style="margin: 0;">Let our bot help you with your queries!</h3>
    </div>
    """, unsafe_allow_html=True)

    # faq_data = load_faq_data()

    # # Add a button to clear the chat history
    # if st.button("Clear Chat History"):
    #     st.session_state.messages = []  # Clear the messages list
    #     st.rerun()  # Force Streamlit to rerun the app

    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # if prompt := st.chat_input("How can I assist you today?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     response = get_response(prompt, faq_data)
    #     with st.chat_message("assistant"):
    #         st.markdown(response)

    #     st.session_state.messages.append({"role": "assistant", "content": response})
        