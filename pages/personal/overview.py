import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Function to format numbers in Indian system
def indian_metric(num):
    if num >= 10_00_00_000:  # 10 crore and above
        return f"₹{num / 10_00_00_000:.2f} Cr"
    elif num >= 1_00_000:  # 1 lakh and above
        return f"₹{num / 1_00_000:.2f} L"
    else:
        return f"₹{num:,.0f}"  # Standard format for smaller numbers


def prep_data(df, freq):
    """
    Aggregates income and expenses based on the selected time frequency.
    :param freq: Resampling frequency ('ME' - Monthly, 'QE' - Quarterly, 'YE' - Yearly).
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Amount'] = df['Amount'].astype(float)

    # Group by selected frequency
    grouped = df.groupby([pd.Grouper(key='Date', freq=freq), 'Income_Expense'])['Amount'].sum().unstack().fillna(0)
    grouped = grouped.rename(columns={'Income': 'Total Income', 'Expense': 'Total Expense'}).reset_index()

    return grouped

def plot_trend(df, show_income=True, show_expense=True):
    """
    Plots  line chart for income and expenses.
    :param show_income: Boolean to toggle income display.
    :param show_expense: Boolean to toggle expense display.
=    """
    fig = go.Figure()

    # Add Income trend line
    if show_income and 'Total Income' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Total Income'],
            mode='lines+markers',
            name='Income',
            line=dict(color='green', width=2, shape='spline'),
            marker=dict(size=5)
        ))

    # Add Expense trend line
    if show_expense and 'Total Expense' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['Total Expense'],
            mode='lines+markers',
            name='Expense',
            line=dict(color='red', width=2, shape='spline'),
            marker=dict(size=5)
        ))

    # Layout settings
    fig.update_layout(
        xaxis=dict(
            showspikes=True,  
            spikemode='across', 
            spikethickness=0.02,
        ),
        title='Income vs. Expense Trend',
        xaxis_title='Date',
        yaxis_title='Amount',
        hovermode='x unified',
        legend=dict(title='Legend', orientation='v', yanchor='top', y=4, xanchor='right', x=4),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

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

def overview():
    """ Financial Snapshot Page"""

    st.markdown(
        """
        <style>
        * {
            font-family: Verdana, sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.write("## 🌐 Overview")
    st.write("A snapshot of your financial metrics reflecting your income and expense trends at a glance.")

    # Initialize session state to store the dataframe
    if 'finance_df' not in st.session_state:
        st.session_state.finance_df = None
        
    if st.session_state.finance_df is None:
        st.warning("Please upload data first, record some transactions or use the sample dataset.")
        return
    else:
        data = st.session_state.finance_df
        df = data.copy()
        
        # Convert Date column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        
        # Add month and year columns for easier grouping
        df['Month'] = df['Date'].dt.month_name()
        df['MonthYear'] = df['Date'].dt.strftime('%b %Y')
        df['Year'] = df['Date'].dt.year
        
        # Identify the latest month and year in the dataset
        latest_date = df['Date'].max()  
        latest_month = latest_date.month
        latest_year = latest_date.year

        # Filter data for the latest month and year
        latest_month_df = df[(df['Date'].dt.month == latest_month) & (df['Date'].dt.year == latest_year)]

        # Create container layout: 3 containers in upper row and 2 in lower row
        upper_row = st.columns(3)
        lower_row = st.columns([2,1])
        
        # Container 1: Balance / Savings Graph
        with upper_row[0]:
            st.markdown("""<h2 style="margin-bottom: 20px; font-size: 27px; font-weight: bold; color: #0b240a; text-align: center; padding: 3px; background-color: #4d9e47; border-radius: 10px;">Balance</h2>""", unsafe_allow_html=True)
            
            # Time frequency selection
            time_options = ["Weekly", "Monthly", "Quarterly", "Yearly"]
            selected_frequency = st.selectbox("Select Time Frequency", time_options, key="balance_frequency")
            
            # Calculate current balance
            income_total = df[df['Income_Expense'] == 'Income']['Amount'].sum()
            expense_total = df[df['Income_Expense'] == 'Expense']['Amount'].sum()
            current_balance = indian_metric(income_total - expense_total)
            
            st.markdown(f"""
                <div style="margin-top: 0px; margin-bottom: -50px;">
                    <span style="font-size: 14px; color: white;">Net Balance at Present</span>
                    <span style="font-size: 38px; color: orange;">{current_balance}</span>
                </div>
            """, unsafe_allow_html=True)

            # Create balance dataframe based on selected frequency
            if selected_frequency == "Weekly":
                df['TimeGroup'] = df['Date'].dt.strftime('%Y-W%U')
                time_col = 'Week'
            elif selected_frequency == "Monthly":
                df['TimeGroup'] = df['Date'].dt.strftime('%b %Y')
                time_col = 'Month'
            elif selected_frequency == "Quarterly":
                df['Quarter'] = df['Date'].dt.year.astype(str) + '-Q' + ((df['Date'].dt.month - 1) // 3 + 1).astype(str)
                df['TimeGroup'] = df['Quarter'] 
                time_col = 'Quarter'
            else:  # Yearly
                df['TimeGroup'] = df['Date'].dt.year
                time_col = 'Year'
            
            # Group by time and calculate balance
            income_by_time = df[df['Income_Expense'] == 'Income'].groupby('TimeGroup')['Amount'].sum().reset_index()
            expense_by_time = df[df['Income_Expense'] == 'Expense'].groupby('TimeGroup')['Amount'].sum().reset_index()
            
            # Create a complete dataframe with all periods
            balance_df = pd.merge(income_by_time, expense_by_time, on='TimeGroup', how='outer').fillna(0)
            balance_df.columns = ['TimeGroup', 'Income', 'Expense']
            balance_df['Balance'] = balance_df['Income'] - balance_df['Expense']
            
            # Sort by date (implicit in TimeGroup)
            if selected_frequency != "Weekly":  # for monthly, quarterly, yearly
                balance_df = balance_df.sort_values('TimeGroup')
            
            # Calculate cumulative balance
            balance_df['Cumulative_Balance'] = balance_df['Balance'].cumsum()
            balance_df['Cumulative_Balance_Lakhs'] = balance_df['Cumulative_Balance'] / 1_00_000
            
            # Create balance line chart
            balance_fig = px.line(
                balance_df, 
                x='TimeGroup', 
                y='Cumulative_Balance_Lakhs',
                markers=True,
                title=f'{selected_frequency} Balance Trend',
            )
            
            balance_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)', 
                xaxis_title='Time Period',  # Custom x-axis label
                yaxis_title='Balance (₹ Lakhs)',  # Custom y-axis label
                height=300
            )
            st.plotly_chart(balance_fig, use_container_width=True)
    
        # Container 2: Income Breakdown
        with upper_row[1]:
            st.markdown("""<h2 style="margin-bottom: 20px; font-size: 27px; font-weight: bold; color: #4d1c2a; text-align: center; padding: 3px; background-color: #eba2b8; border-radius: 10px;">Income</h2>""", unsafe_allow_html=True)

            
            # Calculate current month's income
            latest_month_income = latest_month_df[latest_month_df['Income_Expense'] == 'Income']['Amount'].sum()
            latest_month_income = indian_metric(latest_month_income)

            st.markdown(f"""
                <div style="margin-top: 0px; margin-bottom: -50px;">
                    <span style="font-size: 14px; color: white;">Current Month Inflow</span>
                    <span style="font-size: 38px; color: orange;">{latest_month_income}</span>
                </div>
            """, unsafe_allow_html=True)

            # Income source breakdown
            income_categories = latest_month_df[latest_month_df['Income_Expense'] == 'Income'].groupby('Category')['Amount'].sum().reset_index()
            
            if not income_categories.empty:
                
                # Create pie chart
                income_fig = px.pie(
                    income_categories, 
                    values='Amount', 
                    names='Category',
                    title='Income Sources',
                    hole=0.4
                )
                income_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300)
                st.plotly_chart(income_fig, use_container_width=True)
            else:
                print(len(income_categories))
                st.info("No income transactions recorded for the current month.")
        
        # Container 3: Expenses Breakdown
        with upper_row[2]:
            st.markdown("""<h2 style="margin-bottom: 20px; font-size: 27px; font-weight: bold; color: #152645; text-align: center; padding: 3px; background-color: #6c9aeb; border-radius: 10px;">Expenses</h2>""", unsafe_allow_html=True)

            # Calculate current month's expenses
            latest_month_expenses = latest_month_df[latest_month_df['Income_Expense'] == 'Expense']['Amount'].sum()
            latest_month_expenses = indian_metric(latest_month_expenses)

            st.markdown(f"""
                <div style="margin-top: 0px; margin-bottom: -50px;">
                    <span style="font-size: 14px; color: white;">Current Month Outflow</span>
                    <span style="font-size: 38px; color: orange;">{latest_month_expenses}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Expense category breakdown
            expense_categories = latest_month_df[latest_month_df['Income_Expense'] == 'Expense'].groupby('Category')['Amount'].sum().reset_index()
            
            if not expense_categories.empty:
                # Sort by amount descending
                expense_categories = expense_categories.sort_values('Amount', ascending=True)
                
                # Create horizontal bar chart
                expense_fig = px.bar(
                    expense_categories,
                    y='Category',
                    x='Amount',
                    color='Amount', 
                    color_continuous_scale='viridis', 
                    title='Expense Categories',
                    labels={'Amount': 'Amount (₹)', 'Category': ''}
                )
                expense_fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    coloraxis_showscale=False )
                st.plotly_chart(expense_fig, use_container_width=True)
            else:
                st.info("No expense transactions recorded for the current month.")

        # Container 4: Highlights section
        with lower_row[0]:
            # Calculate the data first
            monthly_expenses = df[df['Income_Expense'] == 'Expense'].groupby('MonthYear')['Amount'].sum().reset_index()
            
            # Find month with highest expenses
            if not monthly_expenses.empty:
                max_expense_month = monthly_expenses.loc[monthly_expenses['Amount'].idxmax()]
                max_month = max_expense_month['MonthYear']
                max_amount = f"₹{max_expense_month['Amount']:,.2f}"
            else:
                max_month = "N/A"
                max_amount = "₹0.00"
            
            # Find largest expense in current month
            if not latest_month_df.empty and 'Expense' in latest_month_df['Income_Expense'].values:
                biggest_purchase = latest_month_df[latest_month_df['Income_Expense'] == 'Expense'].loc[
                    latest_month_df[latest_month_df['Income_Expense'] == 'Expense']['Amount'].idxmax()
                ]
                big_category = biggest_purchase['Category']
                big_amount = f"₹{biggest_purchase['Amount']:,.2f}"
            else:
                big_category = "N/A"
                big_amount = "₹0.00"
            
            st.markdown(f"""
            <div style="background-color: #913942; border-radius: 10px; border: 2px solid white; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <h3 style="color: white; border-bottom: 2px solid white; padding-bottom: 4px;">Highlights</h3>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <div style="width: 48%; background-color: #ed939c; padding: 0px; border-radius: 8px; border-left: 4px solid #521118">
                        <h3 style="color: #521c21; margin-top: 20px;"><center>{max_month}</center></h3>
                        <h5 style="color: #521c21; margin-top: -50px;"><center>{max_amount}</center></h5>
                        <h6 style="color: #612529; margin-top: -10px;"><center>Costliest Month</center></h6>
                    </div>
                    <div style="width: 48%; background-color: #ed939c; padding: 0px; border-radius: 8px; border-left: 4px solid #521118">
                        <h3 style="color: #521c21; margin-top: 20px;"><center>{big_category}</center></h3>
                        <h5 style="color: #521c21; margin-top: -50px;"><center>{big_amount}</center></h5>
                        <h6 style="color: #612529; margin-top: -10px;"><center>Biggest Expense</center></h6>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Container 5: Analytics 
        with lower_row[1]:
            # Filter out unwanted categories
            expense_cats = ["Groceries","Healthcare","Transportation","Utilities","Communication","Education",
                           "Enrichment","Domestic Help","Care Essentials","Financial Dues","Discretionary" ]
            
            income_cats = ["Salary","Side Hustle","Pension","Interest","Rewards","Stocks","Gifts"]
            
            invalid_categories = ["Rent","Miscellaneous"]

            # Filter dataset
            df_analytics = df[~df['Category'].isin(invalid_categories)]

            # Extract latest and previous month
            df_analytics['YearMonth'] = df_analytics['Date'].dt.to_period('M')
            latest_month = df_analytics['YearMonth'].max()
            previous_month = latest_month - 1

            # Group by category and month
            grouped = df_analytics.groupby(['YearMonth', 'Category'])['Amount'].sum().unstack(fill_value=0)

            # Calculate % change
            diff = ((grouped.loc[latest_month] - grouped.loc[previous_month]) / grouped.loc[previous_month]) * 100
        
            # Filter out categories with +inf or -inf
            diff = diff[~diff.isin([float('inf'), float('-inf')])]
            
            # Get top 5 categories with highest absolute percentage change
            top_5_diff = diff.abs().nlargest(5).index

            st.subheader("Analytics")
            for category in top_5_diff:
                change = diff[category]
                if category in expense_cats:
                    color = '#c93640' if change > 0 else '#3c9632'
                elif category in income_cats:
                    color = 'orange' if change < 0 else '#368ec9'
                symbol = '+' if change > 0 else ''
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{symbol}{change:.2f}%</span> {category}", unsafe_allow_html=True)

        # base budget data
        budget_data = {
            "Category":  ["Rent", "Groceries","Healthcare","Transportation","Utilities","Communication",
                        "Education","Enrichment","Domestic Help","Care Essentials","Financial Dues",
                        "Discretionary","Miscellaneous"],
            "Budget": [20000, 15000, 8000, 5000, 4000, 5000, 18000, 8000, 5000, 3000, 25000, 4000, 5000],
            "Icons": ["🏠", "🛒", "💊", "🚗", "💡", "📞", "📚", "🎨", "👩‍🍳", "🧴", "💳", "🎭","🔄"]
        }

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("## 📈 Detailed Trend Analysis")
        
        # Dropdown for time selection
        time_options = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        time_freq = st.selectbox("Select Time Frequency:", list(time_options.keys()))
        selected_freq = time_options[time_freq]
        
        # Checkboxes for income/expense
        show_income = st.checkbox("Show Income", value=True)
        show_expense = st.checkbox("Show Expense", value=True)
        
        # Process Data
        df_grouped = prep_data(df, selected_freq)
        
        # Generate Plot
        fig = plot_trend(df_grouped, show_income, show_expense)
        
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
                hovertemplate='<b>%{label}</b><br>Amount: ₹%{value}'
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