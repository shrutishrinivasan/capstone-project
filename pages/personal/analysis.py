import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# helper functions
def prep_data(df, freq):
    df["Date"] = pd.to_datetime(df["Date"])
    df_grouped = df.groupby([pd.Grouper(key="Date", freq=freq), "Income_Expense"]).sum()["Amount"].unstack(fill_value=0)
    df_grouped = df_grouped.rename(columns={"Expense": "Expenditure", "Income": "Income"})
    return df_grouped

def plot_trend(df_grouped, show_income, show_expense, time_freq):
    fig_data = []
    if show_income:
        fig_data.append("Income")
    if show_expense:
        fig_data.append("Expenditure")
    
    if fig_data:
        fig = px.line(df_grouped[fig_data], 
                      color_discrete_map={'Income': 'green', 'Expenditure': 'red'},
                      markers=True, 
                      title=f"Income & Expenditure Trends ({time_freq})")
        fig.update_layout(xaxis_title="Date", yaxis_title="Amount", legend_title="Type")
        return fig
    return None

def filter_expenses(df, duration, budget_data):
    """Filter the expenses dataframe based on selected duration and prepare data for plotting."""
    
    # Get the end date as the maximum date in the dataset
    end_date = df['Date'].max()
    
    # Determine the start date based on the selected duration
    if duration == "Current Month":
        start_date = end_date.replace(day=1)
    elif duration == "Previous Month":
        start_date = (end_date.replace(day=1) - timedelta(days=1)).replace(day=1)
        end_date = start_date.replace(day=1) + pd.DateOffset(months=1) - timedelta(days=1)
    elif duration == "Last 3 Months":
        start_date = end_date - pd.DateOffset(months=3)
    elif duration == "Last 6 Months":
        start_date = end_date - pd.DateOffset(months=6)
    else:  # Default to "Last 1 Year"
        start_date = end_date - pd.DateOffset(years=1)

    # Filter the DataFrame for expenses within the specified date range
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Group by category and aggregate transactions and amounts
    expense_summary = filtered_df.groupby('Category').agg(
        Transactions=('Transaction_id', 'count'),
        Amount=('Amount', 'sum')
    ).reset_index()
    
    # Merge with budget data
    budget_df = pd.DataFrame(budget_data)
    
    # Adjust budget based on duration
    if duration in ["Current Month", "Previous Month"]:
        expense_summary = expense_summary.merge(budget_df, on='Category', how='left')
    else:
        expense_summary = expense_summary.merge(budget_df, on='Category', how='left')
        expense_summary['Budget'] *= (3 if duration == "Last 3 Months" else 
                                    6 if duration == "Last 6 Months" else 
                                    12)  # For Last 1 Year
    
    return expense_summary

def performance_analysis():
    """Performance Analysis page"""

    st.markdown("""
    <style>
        * {
            font-family: Verdana, sans-serif !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.write("## 📈 Trend Analysis")
    st.write("This section shows detailed analytics about your financial habits and trends to help you improve on your savings.")

    # base budget data
    budget_data = {
        "Category":  ["Rent", "Groceries","Healthcare","Transportation","Utilities","Communication",
                      "Education","Enrichment","Domestic Help","Care Essentials","Financial Dues",
                      "Discretionary","Miscellaneous"],
        "Budget": [20000, 15000, 8000, 5000, 4000, 5000, 18000, 8000, 5000, 3000, 25000, 4000, 5000],
        "Icons": ["🏠", "🛒", "💊", "🚗", "💡", "📞", "📚", "🎨", "👩‍🍳", "🧴", "💳", "🎭","🔄"]
    }

    # Initialize session state to store the dataframe
    if 'finance_df' not in st.session_state:
        st.session_state.finance_df = None
        
    if st.session_state.finance_df is None:
        st.warning("Please upload data first, record some transactions or use the sample dataset.")
        return
    else:
        data = st.session_state.finance_df
        df = data.copy()
        
        st.write("### Income & Expense Trends")
        
        # Dropdown for time selection
        time_options = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        time_freq = st.selectbox("Select Time Frequency:", list(time_options.keys()))
        selected_freq = time_options[time_freq]
        
        # Checkboxes for income/expense
        show_income = st.checkbox("Show Income", value=True)
        show_expense = st.checkbox("Show Expense", value=True)
        
        # Process Data
        df_grouped = prep_data(df, selected_freq)
        
        # Generate Plot
        fig = plot_trend(df_grouped, show_income, show_expense, time_freq)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        
        if fig:
            st.plotly_chart(fig)


        # Expense Distribution Heading
        st.write("### Expense Distribution")

        # Time duration selection
        time_filter = st.selectbox("Select Duration", [
            "Current Month", "Previous Month", "Last 3 Months", "Last 6 Months", "Last 1 Year"
        ])

        if df is None:
            st.warning("Please upload your transaction data first from the 'Upload Data' tab.")
        else:
            filtered_df = filter_expenses(df[df['Income_Expense'] == 'Expense'], duration=time_filter, budget_data=budget_data)

            # Custom Order for Consistent Category Display
            custom_order = [
                "Rent", "Groceries", "Healthcare", "Transportation", "Utilities", 
                "Communication", "Education", "Enrichment", "Domestic Help", 
                "Care Essentials", "Financial Dues", "Discretionary", "Miscellaneous"
            ]

            # Create Custom Color Mapping
            color_palette = px.colors.qualitative.Prism + ['#f2aace', '#66b8c4']
            color_map = dict(zip(custom_order, color_palette[:len(custom_order)]))
            print(color_map)

            fig = px.pie(
                filtered_df, 
                values='Amount', 
                names='Category', 
                hole=0.4,
                category_orders={"Category": custom_order},
                color='Category',
                color_discrete_map=color_map  # Coordinated Colors
            )

            # Custom hover template to include 'Transactions'
            fig.update_traces(
                textinfo='none',
                hovertemplate='<b>%{label}</b><br>Amount: ₹%{value}<br>Transactions: %{customdata[0]}'
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )


            # Display Pie Chart
            st.plotly_chart(fig, use_container_width=True)

            # Ensure 'Category' column follows custom order
            filtered_df['Category'] = pd.Categorical(
                filtered_df['Category'],
                categories=custom_order,
                ordered=True
            )

            # Sort DataFrame by the custom order
            filtered_df = filtered_df.sort_values('Category')

            # Category-wise Expenditure Bars
            st.subheader("Category-wise Breakdown")
            for index, row in filtered_df.iterrows():
                percentage_spent = row["Amount"] / row["Budget"]

                # Coordinated Color Mapping for Bars
                bar_color = color_map.get(row['Category'], '#FFFFFF')

                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 24px; margin-right: 10px;">{row['Icons']}</span>
                        <div style="flex-grow: 1;">
                            <div style="display: flex; justify-content: space-between; font-size: 14px; margin-top: 5px;">
                                <span><b>{row['Category']}</b></span>
                                <span>{row['Transactions']} transactions</span>
                                <span>₹{row['Amount']}</span>
                            </div>
                            <div style="background-color: #34373b; border-radius: 5px; overflow: hidden;">
                                <div style="width: {percentage_spent * 100}%; background-color: {bar_color}; height: 20px;"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # st.write("### Expense Distribution")

        # # Time duration selection
        # time_filter = st.selectbox("Select Duration", ["Current Month", "Previous Month", "Last 3 Months", "Last 6 Months", "Last 1 Year"])

        # if df is None:
        #     st.warning("Please upload your transaction data first from the 'Upload Data' tab.")
        # else:
        #     filtered_df = filter_expenses(df[df['Income_Expense']=='Expense'],duration=time_filter, budget_data=budget_data)
    
        #     custom_order = [
        #         "Rent", "Groceries", "Healthcare", "Transportation", "Utilities", 
        #         "Communication", "Education", "Enrichment", "Domestic Help", 
        #         "Care Essentials", "Financial Dues", "Discretionary", "Miscellaneous"
        #     ]

        #     # Create an interactive pie chart
        #     fig = px.pie(filtered_df, values='Amount', names='Category', hole=0.4,
        #                 category_orders={"Category": filtered_df["Category"].tolist()},
        #                 color_discrete_sequence=px.colors.qualitative.Prism, hover_data={'Transactions': True})

        #     fig.update_traces(textinfo='none', hovertemplate='<b>%{label}</b><br>Amount: ₹%{value}<br>Transactions: %{customdata[0]}')
        #     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')

        #     # Display Pie Chart
        #     st.plotly_chart(fig, use_container_width=True)

        #     # Category-wise Expenditure Bars
        #     st.subheader("Category-wise Breakdown")
        #     for index, row in filtered_df.iterrows():
        #         percentage_spent = row["Amount"] / row["Budget"]
        #         bar_color = px.colors.qualitative.Prism[index % len(px.colors.qualitative.Prism)] 

        #         st.markdown(
        #             f"""
        #             <div style="display: flex; align-items: center; margin-bottom: 10px;">
        #                 <span style="font-size: 24px; margin-right: 10px;">{row['Icons']}</span>
        #                 <div style="flex-grow: 1;">
        #                     <div style="display: flex; justify-content: space-between; font-size: 14px; margin-top: 5px;">
        #                         <span><b>{row['Category']}</b></span>
        #                         <span>{row['Transactions']} transactions</span>
        #                         <span>₹{row['Amount']}</span>
        #                     </div>
        #                     <div style="background-color: #34373b; border-radius: 5px; overflow: hidden;">
        #                         <div style="width: {percentage_spent * 100}%; background-color: {bar_color}; height: 20px;"></div>
        #                     </div>
        #                 </div>
        #             </div>
        #             """,
        #             unsafe_allow_html=True
        #         )
