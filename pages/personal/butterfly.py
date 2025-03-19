import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def butterfly():
    """Butterfly Effect page"""
    
    st.markdown(
        """
        <style>
        * {
            font-family: Verdana, sans-serif !important;
        }
        body {
            background-color: #f0f4f8;
            font-family: Arial, sans-serif;
        }
        .title {
            color: #2a9d8f;
            text-align: center;
            font-size: 40px;
            margin-bottom: 20px;
        }
        .description {
            color: white;
            text-align: center;
            font-size: 50px;
            margin-bottom: 40px;
        }
        .sidebar .sidebar-content {
            background-color: #e9c46a;
        }
        .css-1fcdlhj {
            padding: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="title"> Butterfly Effect Simulator</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="description">Small changes in your daily expenses can create a big impact on your financial future! </p>',
        unsafe_allow_html=True,
    )

    # Sliders for expense categories
    st.sidebar.markdown("**Expense Adjustments**")
    groceries_slider = st.sidebar.slider("🛒Groceries", 0, 3000, 1500, key="groceries")
    transportation_slider = st.sidebar.slider("🚗Transportation", 0, 2000, 1000, key="transportation")
    edu_slider = st.sidebar.slider("🎓Education", 0, 15000, 8000, key="edu")
    healthcare_slider = st.sidebar.slider("⚕️Healthcare", 0, 5000, 2500, key="healthcare")
    utilities_slider = st.sidebar.slider("🛠️Utilities", 0, 1500, 500, key="utilities")
    rent_slider = st.sidebar.slider("🏠Rent", 0, 20000, 9000, key="rent")
    dexp_slider = st.sidebar.slider("💳Discretionary Expenses", 0, 10000, 5000, key="dexp")
    comm_slider = st.sidebar.slider("📞Communication", 0,5000,2500, key="comm")

    # Function to calculate compounded expenses over time with variations
    def calculate_compounded_expenses_with_variations(expenses, months=6):
        compounded_expenses = [expenses]
        for _ in range(months - 1):
            expenses *= np.random.uniform(1.01, 1.03) # Assuming a monthly increase between 1% and 3%
            compounded_expenses.append(expenses)
        return compounded_expenses

    # Calculate total monthly expenses
    original_expenses = groceries_slider + transportation_slider + edu_slider + healthcare_slider + utilities_slider + rent_slider + dexp_slider + comm_slider

    # Calculate adjusted expenses based on slider values
    adjusted_expenses = groceries_slider + transportation_slider + edu_slider + healthcare_slider + utilities_slider + rent_slider + dexp_slider + comm_slider

    predicted_expenses = calculate_compounded_expenses_with_variations(adjusted_expenses)

    # Layout: Two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍Insights")
        st.markdown(
            """
        Adjust the sliders to adjust expense categories and explore further.
            """
        )
        st.info(
            "💡 **Tip:** Reduce discretionary spending by even 5% to see significant long-term savings!"
        )
        st.markdown("**Suggested Adjustments:**")
        st.write(
            """
            - 💸 Save 10% on rent to free up ₹12,000 annually for investments.
            - 🚗 Reduce transportation costs by carpooling to save ₹5,000/year.
            """
        )

    # Line chart with variations
    with col2:
        x = np.arange(1, 7)

        # Adding slight variations to the original expenses
        y_original = calculate_compounded_expenses_with_variations(original_expenses)

        # Adding more steep peaks and valleys to the predicted expenses
        y_predicted = [expense * np.random.uniform(1.02, 1.05) for expense in y_original]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_original, mode='lines+markers', name="Original Expenses"))
        fig.add_trace(go.Scatter(x=x, y=y_predicted, mode='lines+markers', name="Predicted Expenses", line=dict(dash='dot')))

        fig.update_layout(
            title="Impact of Minor Changes",
            title_font=dict(size=24),
            font=dict(
                color="white"
            ),
            xaxis=dict(
                title="Months",
                title_font=dict(color="white",size=15),
                tickfont=dict(color="white",size=15)
            ),
            yaxis=dict(
                title="Expenditure (INR)",
                title_font=dict(color="white",size=15),
                tickfont=dict(color="white",size=15)
            ),
            template="plotly_dark",
            # margin=dict(l=40, r=40, t=40, b=40),
            width=4500,
            height=500,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
        )

        st.plotly_chart(fig)

    # Horizontal bar chart with variations
    categories = ["Groceries", "Transportation", "Education", "Healthcare", "Utilities", "Rent", "Discretionary Expenses", "Communication"]
    actual_spending = [groceries_slider, transportation_slider, edu_slider, healthcare_slider, utilities_slider, rent_slider, dexp_slider, comm_slider]

    # Adjusted spending based on slider values
    adjusted_spending = []
    for actual, slider in zip(actual_spending, [groceries_slider, transportation_slider, edu_slider, healthcare_slider, utilities_slider, rent_slider, dexp_slider, comm_slider]):
        if slider > actual:
            adjusted_spending.append(actual * np.random.uniform(1.02, 1.05))
        else:
            adjusted_spending.append(actual * np.random.uniform(0.95, 0.98))

    df = pd.DataFrame({"Category": categories, "Actual": actual_spending, "Adjusted": adjusted_spending})
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(y=df["Category"], x=df["Actual"], name="Actual", orientation="h",marker=dict(color="#a9084f")))
    fig2.add_trace(go.Bar(y=df["Category"], x=df["Adjusted"], name="Adjusted", orientation="h",marker=dict(color="#28a546 ")))

    fig2.update_layout(
        font=dict(
            color="white"
        ),
        xaxis=dict(
            title="Amount (INR)",
            title_font=dict(color="white",size=15),
            tickfont=dict(color="white",size=15)
        ),
        yaxis=dict(
            title="Expense Categories",
            title_font=dict(color="white",size=15),
            tickfont=dict(color="white",size=15)
        ),
        title="Adjusted Budget vs. Actual Spending",
        title_font=dict(size=24),
        template="plotly_dark",
        height=400,
        barmode="group",
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )

    st.plotly_chart(fig2)

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center;">💡 Small changes today can lead to big savings tomorrow!</p>',
        unsafe_allow_html=True,
    )

    if st.button("⬅ Back to Dashboard"):
                st.session_state.menu_state = "main"
                st.rerun()