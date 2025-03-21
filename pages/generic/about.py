import streamlit as st
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

encoded_logo = get_base64_image(r"C:\Users\USER\Desktop\capstone_project\static\logo.png")


def about_page():
    """About section of the landing page"""
    
    # Set the font for the entire page
    st.markdown("""
    <style>
    * {
        font-family: Verdana, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Container with light background for the title and logo - made smaller with less padding
    with st.container():
        st.markdown(f"""
        <div style="background-color: #fffdd0; padding: 0px 100px; border-radius: 20px; width: 90%; margin-left: 70px; margin-bottom: 30px">
            <div style="display: flex; align-items: center;">
                <div style="flex: 3;">
                    <h2 style="margin: 0; color: #1E1E1E;">Your Money, Sorted.<br>Smart Finance, Simplified.</h2>
                </div>
                <div style="flex: 1; text-align: right;">
                    <img src="data:image/png;base64,{encoded_logo}" alt="Finance App Logo" width="160">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; align-items: center;">
        <h3 style="margin: 0;">What is PaisaVault all about?</h3>
        <div style="flex-grow: 1; height: 2px; background-color: white;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="text-align: center;">
        <h4>🔒 Privacy-First Approach</h4>
        <ul style="display: inline-block; text-align: left;">
            <li>No bank account linking required</li>
            <li>Upload data, categorize transactions</li>
            <li>Your data stays yours</li>
        </ul>
        
        <h4>📊 Smart Visualization</h4>
        <ul style="display: inline-block; text-align: left;">
            <li>Clear financial breakdowns</li>
            <li>Interactive charts and graphs</li>
            <li>Spot trends at a glance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="text-align: center;">
        <h4>🤖 AI-Powered Insights</h4>
        <ul style="display: inline-block; text-align: left;">
            <li>Personalized recommendations</li>
            <li>Predictive spending analysis</li>
            <li>Custom financial plans</li>
        </ul>
        
        <h4>🎯 Goal Setting & Tracking</h4>
        <ul style="display: inline-block; text-align: left;">
            <li>Set and visualize financial goals</li>
            <li>Track progress in real-time</li>
            <li>Achieve long term plans</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional features
    st.markdown("""
    <div style="display: flex; align-items: center; margin-top: 30px">
        <h3 style="margin: 0;">More than Just Budgeting</h3>
        <div style="flex-grow: 1; height: 2px; background-color: white;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("""
        <div style="text-align: center;">
        <h3>💬 AI Assistant</h3>
        <p>Get guidance from models trained on your data</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div style="text-align: center;">
        <h3>🔮 Financial Simulations</h3>
        <p>See how today's decisions impact tomorrow's finances</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown("""
        <div style="text-align: center;">
        <h3>📰 Financial Education</h3>
        <p>Learn as you go with tailored resources</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action with darker text and nicer background
    st.markdown("""
    <div style="background-color: #e6f2ff; padding: 10px; border-radius: 20px; margin-top: 30px; text-align: center;">
        <h2 style="color: #003366;">Ready to take control of your finances?</h2>
        <p style="color: #0d47a1; font-weight: 1000;">Start your journey to financial wellness today.</p>
    </div>
    """, unsafe_allow_html=True)