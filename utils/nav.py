import streamlit as st
from streamlit_option_menu import option_menu

def horizontal_navbar():
    """Create horizontal navbar for landing page"""
    selected = option_menu(
        menu_title=None,
        options=["About", "Features", "Tools", "Bot", "Learn", "Log In"],
        icons=["info-circle", "list-check", "tools", "robot", "book", "box-arrow-in-right"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#333"},
            "icon": {"color": "white", "font-size": "14px"},
            "nav-link": {
                "font-family": 'Verdana',
                "font-size": "15px",
                "text-align": "center",
                "margin": "8px",
                "--hover-color": "#444",
            },
            "nav-link-selected": {"background-color": "#1E1E24"},
        }
    )
    return selected

def vertical_navbar():
    """Create vertical navbar for dashboard"""
    with st.sidebar:
        selected = option_menu(
            menu_title="Finance Manager",
            options=[
                "Getting Started",
                "Upload Data",
                "Overview",
                "Income/Expense",
                "Goals & Savings",
                "Custom Bot",
                "Performance Analysis",
                "Financial Suggestions",
                "Education",
                "Settings",
                "Log Out"
            ],
            icons=[
                "house",
                "cloud-upload",
                "pie-chart",
                "currency-dollar",
                "piggy-bank",
                "chat-dots",
                "activity",
                "lightbulb",
                "mortarboard",
                "gear",
                "box-arrow-right"
            ],
            menu_icon="bank",
            default_index=0,
            styles={
                "container": {"padding": "4!important", "background-color": "#262730"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-family": "Candara, sans-serif",
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#3A3B46",
                },
                "nav-link-selected": {"background-color": "#1E1E24"},
                "menu-title": {
                    "font-family": "Candara, sans-serif",
                    "font-size": "24px",
                    "font-weight": "bold",
                }
            }
        )
        
        # Display username in sidebar
        if 'username' in st.session_state:
            st.markdown(f"**Logged in as:** {st.session_state.username}")
            
    return selected

def sub_menu():
    """Create submenu for financial suggestions"""
    with st.sidebar:
        selected = option_menu(
            menu_title="Financial Suggestions",
            options=[
                "Butterfly Effect",
                "Scenario Testing",
            ],
            icons=[
                "wind",
                "graph-up-arrow",
            ],
            menu_icon="lightbulb",
            default_index=0,
            styles={
                "container": {"padding": "4!important", "background-color": "#262730"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-family": "Candara, sans-serif",
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#3A3B46",
                },
                "nav-link-selected": {"background-color": "#1E1E24"},
                "menu-title": {
                    "font-family": "Candara, sans-serif",
                    "font-size": "20px",
                    "font-weight": "bold",
                }
            }
        )
    
    return selected