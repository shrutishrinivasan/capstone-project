import streamlit as st 
import plotly.graph_objects as go
from streamlit_extras.stylable_container import stylable_container

def calculate_compound_interest(principal, rate, time, compounds_per_year):
    """Function to calculate compound interest"""
    if principal == 0 or rate == 0 or time == 0 or compounds_per_year == 0:
        return "Please enter valid values for all fields."
    else:
        r = rate / 100
        amount = principal * (1 + r / compounds_per_year) ** (compounds_per_year * time)
        compound_interest = amount - principal
        return round(compound_interest, 2)
  
def calculate_simple_interest(principal, rate, time):
    """Function to calculate simple interest"""
    return (principal * rate * time) / 100

def calculate_total_amount(principal, interest):
    """Function to calculate total amount"""
    return principal + interest

def tools_page():
    """Tools section of the landing page"""

    # CSS
    st.markdown(
        """
        <style>
        * {
            font-family: Verdana, sans-serif !important;
        }
        .result-container {
            padding: 5px;
            border-radius: 5px;
            margin-bottom: 3px;
            text-align: center;
            font-size: 1.6em;  /* Reduced size */
        }
        .slider-label {
            font-size: 1.2em;
            margin-bottom: 0.2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h3 style='margin-bottom: -10px;'>Try out some of our handy tools...</h3>", unsafe_allow_html=True)

    st.markdown("<h4>EMI Calculator</h4>", unsafe_allow_html=True)  

    col3, col1, col2 = st.columns([0.2, 1.5, 2])

    with col1:
        with st.container():
            total_amount_output = st.markdown("<div class='result-container'><span style='color: orange; font-weight: bold; font-size: 1.4em;'>₹0</span></div>", unsafe_allow_html=True)

        with st.container():
            chart_placeholder = st.empty()

    with col2:
        st.markdown("<div class='slider-label'>Loan Amount (₹)</div>", unsafe_allow_html=True)
        with st.columns([0.2, 1, 0.2])[1]:
            principal = st.slider("", min_value=0, max_value=1000000, value=1000, step=1, key="loan_amt")

        st.markdown("<div class='slider-label'>Rate of Interest (% p.a.)</div>", unsafe_allow_html=True)
        with st.columns([0.2, 1, 0.2])[1]:
            rate = st.slider("", min_value=0.0, max_value=30.0, value=3.0, step=0.1, key="interest")

        st.markdown("<div class='slider-label'>Loan Tenure (Years)</div>", unsafe_allow_html=True)
        with st.columns([0.2, 1, 0.2])[1]:
            time = st.slider("", min_value=0, max_value=30, value=5, step=1, key="loan_emi_time")

        interest = calculate_simple_interest(principal, rate, time)
        total_amount = calculate_total_amount(principal, interest)

        total_amount_output.markdown(f"<div class='result-container'><span style='color: orange; font-weight: bold; font-size: 1.4em;'>₹{total_amount:,.2f}</span></div>", unsafe_allow_html=True)

        fig = go.Figure(data=[go.Pie(labels=["Principal", "Interest"], values=[principal, interest], hole=.3)])
        fig.update_layout(
            title_text="Loan Breakdown", 
            title_x=0.4,  # Center title
            title_y=0.9,
            title_xanchor="center",
            title_yanchor="top",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        chart_placeholder.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        with stylable_container(
            key="simple_interest_style",
            css_styles="""
                {
                    background-color: #efebef;
                    color: black;
                    border-radius: 15px;
                    padding: 20px; 
                    text-align: center; 
                    width: 400px;
                    margin-left: 100px;
                }
            """,
        ):
            st.markdown("""<h3 style='margin-left: -140px'>S. I. Calculator</h3>""", unsafe_allow_html=True,)

            with stylable_container(
                key="si_result_container",
                css_styles="""
                    {
                        background-color: #D6FF58; 
                        padding: 8px; 
                        border-radius: 5px; 
                        margin-bottom: 10px; 
                        height: 40px;
                        width: 340px;
                    }
                """,
            ):
                si_result_container = st.markdown("₹\nTotal Simple Interest") 

            st.markdown(
                                    """
                                    <style>
                                    label {
                                        color: #4A4A4A !important;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True
                                )
            with st.columns([0.3, 2.5, 0.3])[1]:  
                principal = st.number_input("Principal Amount (₹)", min_value=0.0, key="si_p")
            with st.columns([0.3, 2.5, 0.3])[1]: 
                rate = st.number_input("Rate of Interest (% p.a.)", min_value=0.0, key="si_rate")
            with st.columns([0.3, 2.5, 0.3])[1]: 
                time = st.number_input("Time Period (Years)", min_value=0.0, key="si_time")

            with stylable_container(
                key="si_button",
                css_styles="""
                    button {
                        margin-top: 30px;
                        background-color: black;
                        color: white;
                        border-radius: 15px;
                        padding: 8px 15px; 
                    }
                """,
            ):
                if st.button("Calculate", key="si_button"):
                    result = calculate_simple_interest(principal, rate, time)
                    si_result_container.write(f"₹{result}  \nTotal Simple Interest")

    with col2:   
        with stylable_container(
            key="compound_interest_style",
            css_styles="""
                {
                    background-color: #efebef;
                    color: black;
                    border-radius: 15px;
                    padding: 15px; 
                    text-align: center; 
                    width: 400px;
                    margin-left: 50px;
                }
            """,
        ):
            
            st.markdown("""<h3 style='margin-left: -110px'>C. I. Calculator</h3>""", unsafe_allow_html=True,)

            with stylable_container(
                key="ci_result_container",
                css_styles="""
                    {
                        background-color: #D6FF58; 
                        padding: 8px; 
                        border-radius: 5px; 
                        margin-bottom: 10px; 
                        height: 40px;
                        color: black;
                    }
                """,
            ):
                ci_result_container = st.markdown("₹\nTotal Compound Interest") 

            st.markdown(
                        """
                        <style>
                        label {
                            color: #4A4A4A !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
            with st.columns([0.3, 2.5, 0.3])[1]:  
                principal = st.number_input("Principal Amount (₹)", min_value=0.0, key="ci_p")

            with st.columns([0.3, 2.5, 0.3])[1]: 
                rate = st.number_input("Rate of Interest (% p.a.)", min_value=0.0, key="ci_rate")

            with st.columns([0.3, 2.5, 0.3])[1]:  
                time = st.number_input("Time Period (Years)", min_value=0.0, key="ci_time")

            with st.columns([0.3, 2.5, 0.3])[1]: 
                compounds_per_year = st.number_input("Compounds per Year", min_value=1, key="ci_year")

            with stylable_container(
                key="ci_button",
                css_styles="""
                    button {
                        background-color: black;
                        color: white;
                        border-radius: 15px;
                        padding: 8px 15px; 
                        margin-top: 30px;
                    }
                """,
            ):
                if st.button("Calculate", key="ci_button"):
                    result = calculate_compound_interest(principal, rate, time, compounds_per_year)
                    ci_result_container.write(f"₹{result}  \nTotal Compound Interest")