import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def scenario():
    """Scenario Tester page"""

    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #4ECDC4;
            margin-bottom: 1rem;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #edf2f7;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #718096;
        }
        .highlight {
            color: #FF6B6B;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>Financial Scenario Tester</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Test your financial resilience and plan for your financial goals</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Your Financial Profile")
        
        # Personal Information
        age = st.slider("Age", 18, 75, 30)
        dependents = st.number_input("Number of Dependents", 0, 10, 1)
        
        # Income Details
        st.subheader("Income Details")
        monthly_income = st.number_input("Monthly Income (₹)", 10000, 1000000, 50000)
        additional_income = st.number_input("Additional Monthly Income (₹)", 0, 500000, 5000)
        
        # Expense Details
        st.subheader("Monthly Expenses")
        rent = st.number_input("Rent/Home Loan EMI (₹)", 0, 200000, 15000)
        utilities = st.number_input("Utilities (₹)", 1000, 50000, 5000)
        groceries = st.number_input("Groceries (₹)", 1000, 50000, 8000)
        transport = st.number_input("Transportation (₹)", 500, 20000, 3000)
        education = st.number_input("Education (₹)", 0, 100000, 2000)
        entertainment = st.number_input("Entertainment & Leisure (₹)", 0, 50000, 3000)
        
        # Existing Assets
        st.subheader("Your Assets")
        emergency_fund = st.number_input("Emergency Fund (₹)", 0, 10000000, 100000)
        stocks_value = st.number_input("Stocks/Mutual Funds Value (₹)", 0, 10000000, 200000)
        fd_value = st.number_input("Fixed Deposits (₹)", 0, 10000000, 150000)
        gold_value = st.number_input("Gold Value (₹)", 0, 10000000, 50000)
        real_estate_value = st.number_input("Real Estate Value (₹)", 0, 50000000, 2000000)
        ppf_epf = st.number_input("PPF/EPF Balance (₹)", 0, 10000000, 100000)
        
        # Existing Liabilities
        st.subheader("Your Liabilities")
        home_loan = st.number_input("Home Loan Outstanding (₹)", 0, 50000000, 2000000)
        car_loan = st.number_input("Car Loan Outstanding (₹)", 0, 10000000, 300000)
        personal_loan = st.number_input("Personal Loan Outstanding (₹)", 0, 10000000, 100000)
        credit_card_debt = st.number_input("Credit Card Debt (₹)", 0, 1000000, 20000)
        education_loan = st.number_input("Education Loan (₹)", 0, 10000000, 0)

    # Calculate key financial metrics
    total_monthly_income = monthly_income + additional_income
    total_monthly_expenses = rent + utilities + groceries + transport + education + entertainment
    monthly_savings = total_monthly_income - total_monthly_expenses
    annual_savings = monthly_savings * 12

    total_assets = emergency_fund + stocks_value + fd_value + gold_value + real_estate_value + ppf_epf
    total_liabilities = home_loan + car_loan + personal_loan + credit_card_debt + education_loan
    net_worth = total_assets - total_liabilities

    # Debt to income ratio
    if total_monthly_income > 0:
        debt_to_income_ratio = ((home_loan + car_loan + personal_loan + credit_card_debt + education_loan) * 0.01) / (total_monthly_income * 12) * 100
    else:
        debt_to_income_ratio = 0

    # Emergency fund ratio (in months)
    if total_monthly_expenses > 0:
        emergency_fund_months = emergency_fund / total_monthly_expenses
    else:
        emergency_fund_months = 0

    # Main dashboard
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 class='sub-header'>Your Financial Snapshot</h2>", unsafe_allow_html=True)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.markdown(f"""
            <div style='background-color: #2D3142 ,class='metric-card'>
                <div class='metric-value'>₹{net_worth:,.0f}</div>
                <div class='metric-label'>Net Worth</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f"""
            <div style='background-color: #2D3142 class='metric-card'>
                <div class='metric-value'>₹{monthly_savings:,.0f}</div>
                <div class='metric-label'>Monthly Savings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            st.markdown(f"""
            <div style='background-color: #2D3142 class='metric-card'>
                <div class='metric-value'>{emergency_fund_months:.1f}</div>
                <div class='metric-label'>Emergency Fund (Months)</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Income vs Expenses Chart
        st.markdown("<h3>Income vs Expenses Breakdown</h3>", unsafe_allow_html=True)
        
        # Create income breakdown
        income_data = {'Category': ['Primary Income', 'Additional Income'], 
                    'Amount': [monthly_income, additional_income]}
        income_df = pd.DataFrame(income_data)
        
        # Create expense breakdown
        expense_data = {'Category': ['Rent/EMI', 'Utilities', 'Groceries', 'Transport', 'Education', 'Entertainment'],
                    'Amount': [rent, utilities, groceries, transport, education, entertainment]}
        expense_df = pd.DataFrame(expense_data)
        
        # Plot income vs expenses
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=income_df['Category'],
            y=income_df['Amount'],
            name='Income',
            marker_color='#4ECDC4'
        ))
        
        fig.add_trace(go.Bar(
            x=expense_df['Category'],
            y=expense_df['Amount'],
            name='Expenses',
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title='Monthly Income vs Expenses',
            xaxis_title='Category',
            yaxis_title='Amount (₹)',
            barmode='group',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Asset Allocation Pie Chart
        st.markdown("<h3>Your Asset Allocation</h3>", unsafe_allow_html=True)
        
        asset_data = {
            'Asset': ['Emergency Fund', 'Stocks/MFs', 'Fixed Deposits', 'Gold', 'Real Estate', 'PPF/EPF'],
            'Value': [emergency_fund, stocks_value, fd_value, gold_value, real_estate_value, ppf_epf]
        }
        asset_df = pd.DataFrame(asset_data)
        
        fig = px.pie(
            asset_df,
            values='Value',
            names='Asset',
            title='Asset Allocation',
            color_discrete_sequence=px.colors.sequential.Agsunset_r,
            hole=0.4
        )
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<h2 class='sub-header'>Financial Scenario Testing</h2>", unsafe_allow_html=True)
        
        # Scenario Selection
        scenario = st.selectbox(
            "Select a scenario to test your financial resilience:",
            ["Job Loss", "Medical Emergency", "Market Crash", "High Inflation"]
        )
        
        # Job Loss Scenario
        if scenario == "Job Loss":
            st.markdown("<h3>Job Loss Scenario</h3>", unsafe_allow_html=True)
            
            unemployment_duration = st.slider("Expected months without job", 1, 24, 6)
            severance_pay = st.number_input("Expected severance pay (₹)", 0, 5000000, monthly_income*2)
            reduced_expenses = st.slider("Expense reduction during unemployment (%)", 0, 50, 20)
            
            # Calculate scenario metrics
            adjusted_monthly_expenses = total_monthly_expenses * (1 - reduced_expenses/100)
            available_funds = emergency_fund + severance_pay + (stocks_value * 0.8)  # Assuming 20% loss on liquidating stocks
            
            survival_duration = available_funds / adjusted_monthly_expenses if adjusted_monthly_expenses > 0 else float('inf')
            
            # Calculate survivability score
            if survival_duration >= unemployment_duration:
                survivability_score = min(100, 75 + (survival_duration - unemployment_duration) * 5)
            else:
                survivability_score = max(10, (survival_duration / unemployment_duration) * 75)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Months You Can Survive", f"{survival_duration:.1f}")
                
                # Create survivability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=survivability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Survivability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF6B6B"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': survivability_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  height=250, 
                                  margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cash flow projection
                months = list(range(1, unemployment_duration + 1))
                remaining_funds = [max(0, available_funds - (adjusted_monthly_expenses * m)) for m in months]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=months,
                    y=remaining_funds,
                    mode='lines+markers',
                    name='Remaining Funds',
                    line=dict(color='#4ECDC4', width=3)
                ))
                
                fig.add_shape(
                    type="line",
                    x0=1,
                    y0=0,
                    x1=unemployment_duration,
                    y1=0,
                    line=dict(color="Red", width=2, dash="dash"),
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title="Fund Depletion Projection",
                    xaxis_title="Month",
                    yaxis_title="Remaining Funds (₹)",
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Financial suggestions
            st.markdown("<h4>Financial Suggestions</h4>", unsafe_allow_html=True)
            if survivability_score >= 80:
                st.markdown("""
                ✅ **Strong Position**: Your emergency fund and liquid assets are sufficient for this scenario.
                
                **Recommendations:**
                - Continue maintaining your emergency fund of at least 6 months of expenses
                - Consider investing excess emergency funds into liquid mutual funds for better returns
                - Keep your resume and professional network updated as a precaution
                """)
            elif survivability_score >= 50:
                st.markdown("""
                🟡 **Moderately Prepared**: You can survive the scenario but with potential challenges.
                
                **Recommendations:**
                - Increase your emergency fund to cover at least {:.1f} months of expenses
                - Identify non-essential expenses that can be immediately cut if needed
                - Consider supplemental income sources like freelancing or part-time work
                - Maintain a good credit score for emergency loans if absolutely necessary
                """.format(max(unemployment_duration, 6)))
            else:
                st.markdown("""
                🔴 **Vulnerable Position**: Your current financial situation puts you at significant risk.
                
                **Urgent Recommendations:**
                - Immediately begin building your emergency fund - aim for at least 3 months of expenses
                - Develop multiple income streams (side gigs, freelancing, etc.)
                - Consider skills development to improve job marketability
                - Identify all expenses that can be reduced or eliminated in an emergency
                - Check eligibility for government assistance programs
                - Consider income protection insurance
                """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Medical Emergency Scenario
        elif scenario == "Medical Emergency":
            st.markdown("<h3>Medical Emergency Scenario</h3>", unsafe_allow_html=True)
            
            medical_expense = st.slider("Potential medical expense (₹)", 50000, 2000000, 500000)
            insurance_coverage = st.number_input("Health insurance coverage (₹)", 0, 5000000, 300000)
            recovery_time = st.slider("Recovery time (months)", 1, 12, 3)
            income_reduction = st.slider("Income reduction during recovery (%)", 0, 100, 30)
            
            # Calculate scenario metrics
            out_of_pocket = max(0, medical_expense - insurance_coverage)
            reduced_income = total_monthly_income * (1 - income_reduction/100)
            income_loss = (total_monthly_income - reduced_income) * recovery_time
            total_financial_impact = out_of_pocket + income_loss
            
            available_funds = emergency_fund + fd_value * 0.95  # 5% penalty for early FD withdrawal
            
            # Calculate survivability score
            coverage_ratio = available_funds / total_financial_impact if total_financial_impact > 0 else 1
            survivability_score = min(100, max(10, coverage_ratio * 80))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Financial Impact", f"₹{total_financial_impact:,.0f}")
                st.metric("Available Emergency Funds", f"₹{available_funds:,.0f}")
                
                # Create survivability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=survivability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Financial Resilience Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF6B6B"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': survivability_score
                        }
                    }
                ))
                
                fig.update_layout(height=250, 
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Medical expense breakdown
                fig = go.Figure()
                
                labels = ["Insurance Coverage", "Out of Pocket", "Income Loss"]
                values = [min(insurance_coverage, medical_expense), out_of_pocket, income_loss]
                
                fig = px.pie(
                    names=labels,
                    values=values,
                    title="Medical Emergency Financial Impact",
                    color_discrete_sequence=['#4ECDC4', '#FF6B6B', '#F9C74F'],
                    hole=0.4
                )
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  height=250, 
                                  margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Financial suggestions
            st.markdown("<h4>Financial Suggestions</h4>", unsafe_allow_html=True)
            if survivability_score >= 80:
                st.markdown("""
                ✅ **Well Protected**: Your insurance coverage and emergency fund are adequate.
                
                **Recommendations:**
                - Consider a super top-up health insurance policy for catastrophic coverage
                - Ensure you have critical illness and personal accident coverage
                - Maintain your emergency fund and health insurance premiums
                - Consider creating a separate medical emergency fund for co-pays and deductibles
                """)
            elif survivability_score >= 50:
                st.markdown("""
                🟡 **Moderately Prepared**: You have some protection but could face financial strain.
                
                **Recommendations:**
                - Increase your health insurance coverage to at least ₹{:,.0f}
                - Build a dedicated medical emergency fund of at least ₹1,00,000
                - Consider a family floater policy to cover dependents
                - Research affordable healthcare facilities in your area
                - Review your insurance policy exclusions and waiting periods
                """.format(medical_expense * 1.2))
            else:
                st.markdown("""
                🔴 **Highly Vulnerable**: A medical emergency would cause severe financial distress.
                
                **Urgent Recommendations:**
                - Immediately purchase or upgrade health insurance coverage
                - Consider an affordable term insurance policy with critical illness rider
                - Research government health schemes like Ayushman Bharat if eligible
                - Save at least ₹50,000 in a dedicated medical emergency fund
                - Research low-cost healthcare options in your area
                - Discuss payment plans with hospitals in advance
                """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Market Crash Scenario
        elif scenario == "Market Crash":
            st.markdown("<h3>Market Crash Scenario</h3>", unsafe_allow_html=True)
            
            market_decline = st.slider("Market decline severity (%)", 10, 60, 30)
            duration_years = st.slider("Expected recovery time (years)", 1, 10, 3)
            gold_change = st.slider("Gold price change (%)", -20, 40, 15)
            real_estate_change = st.slider("Real estate price change (%)", -30, 10, -10)
            
            # Calculate scenario impact
            stocks_impact = stocks_value * (market_decline / 100)
            gold_impact = gold_value * (gold_change / 100)
            real_estate_impact = real_estate_value * (real_estate_change / 100)
            
            total_portfolio_change = -stocks_impact + gold_impact + real_estate_impact
            portfolio_percent_change = (total_portfolio_change / total_assets) * 100 if total_assets > 0 else 0
            
            # Calculate recovery trajectory
            years = list(range(0, duration_years+1))
            recovery_values = []
            
            # Initial impact
            new_portfolio_value = total_assets + total_portfolio_change
            recovery_values.append(new_portfolio_value)
            
            # Recovery trajectory assuming annual growth after crash
            for i in range(1, duration_years+1):
                recovery_rate = 0.12  # Assuming 12% average annual return during recovery
                new_portfolio_value = new_portfolio_value * (1 + recovery_rate)
                recovery_values.append(new_portfolio_value)
            
            # Calculate resilience score
            if portfolio_percent_change >= 0:
                resilience_score = 100
            else:
                # How long until portfolio recovers original value
                recovery_years = 0
                for i, value in enumerate(recovery_values):
                    if value >= total_assets:
                        recovery_years = i
                        break
                
                if recovery_years == 0:
                    resilience_score = max(40, 100 + portfolio_percent_change)
                else:
                    resilience_score = max(20, min(90, 100 - (recovery_years/duration_years * 30) - abs(portfolio_percent_change)/2))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Portfolio Impact", 
                    f"₹{total_portfolio_change:,.0f}", 
                    f"{portfolio_percent_change:.1f}%", 
                    delta_color="inverse"
                )
                
                # Create resilience gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=resilience_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Portfolio Resilience Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF6B6B"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': resilience_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Portfolio recovery projection
                fig = go.Figure()
                
                # Original value line
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=total_assets,
                    x1=duration_years,
                    y1=total_assets,
                    line=dict(color="green", width=2, dash="dash"),
                )
                
                # Recovery trajectory
                fig.add_trace(go.Scatter(
                    x=years,
                    y=recovery_values,
                    mode='lines+markers',
                    name='Portfolio Value',
                    line=dict(color='#4ECDC4', width=3)
                ))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title="Portfolio Recovery Projection",
                    xaxis_title="Years",
                    yaxis_title="Value (₹)",
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Financial suggestions
            st.markdown("<h4>Financial Suggestions</h4>", unsafe_allow_html=True)
            if resilience_score >= 80:
                st.markdown("""
                ✅ **Well Diversified**: Your portfolio shows good resilience to market volatility.
                
                **Recommendations:**
                - Continue maintaining a diversified asset allocation
                - Consider this market decline an opportunity to buy quality stocks at discount
                - Maintain 5-10% cash reserves for strategic investing during downturns
                - Review your asset allocation annually to ensure proper diversification
                - Stay invested and avoid panic selling during market crashes
                """)
            elif resilience_score >= 50:
                st.markdown("""
                🟡 **Moderately Resilient**: Your portfolio has some vulnerability to market fluctuations.
                
                **Recommendations:**
                - Increase diversification across asset classes (debt, equity, gold, real estate)
                - Consider increasing debt allocation to 40% of portfolio for stability
                - Implement systematic investment plans (SIPs) to benefit from rupee cost averaging
                - Build a strategic cash reserve of 15-20% to deploy during market corrections
                - Review and potentially rebalance your portfolio quarterly
                """)
            else:
                st.markdown("""
                🔴 **Vulnerable Portfolio**: Your investments are at high risk during market downturns.
                
                **Urgent Recommendations:**
                - Immediately diversify your portfolio across multiple asset classes
                - Increase allocation to fixed income securities (FDs, govt bonds, debt funds)
                - Consider investing 10-15% in gold as a hedge against market volatility
                - Implement asset allocation based on your age and risk tolerance (100 minus age rule for equity exposure)
                - Consider consulting a financial advisor for portfolio restructuring
                - Avoid lump sum investments in the current scenario; use STP (Systematic Transfer Plan) instead
                """)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # High Inflation Scenario
        elif scenario == "High Inflation":
            st.markdown("<h3>High Inflation Scenario</h3>", unsafe_allow_html=True)
            
            inflation_rate = st.slider("Annual inflation rate (%)", 6, 20, 10)
            duration_years = st.slider("Duration of high inflation (years)", 1, 5, 2)
            income_growth_rate = st.slider("Expected annual income growth (%)", 0, 15, 6)
            
            # Calculate scenario impact
            current_expenses = total_monthly_expenses * 12  # Annual expenses
            current_income = total_monthly_income * 12  # Annual income
            
            # Project expenses and income over time
            years = list(range(0, duration_years+1))
            expense_projection = [current_expenses * ((1 + inflation_rate/100) ** y) for y in years]
            income_projection = [current_income * ((1 + income_growth_rate/100) ** y) for y in years]
            
            # Calculate purchasing power impact
            purchasing_power_loss = ((1 - (1 / ((1 + inflation_rate/100) ** duration_years))) * 100)
            
            # Calculate real returns on investments
            fd_real_return = 5.5 - inflation_rate  # Assuming 5.5% FD rate
            ppf_real_return = 7.1 - inflation_rate  # Assuming 7.1% PPF rate
            equity_real_return = 12 - inflation_rate  # Assuming 12% equity return
            
            # Calculate inflation resilience score
            income_expense_gap = sum([(i-e)/i for i, e in zip(income_projection, expense_projection)]) / len(income_projection)
            real_return_factor = (fd_real_return * fd_value + ppf_real_return * ppf_epf + equity_real_return * stocks_value) / (fd_value + ppf_epf + stocks_value) if (fd_value + ppf_epf + stocks_value) > 0 else -inflation_rate
            
            resilience_score = max(10, min(100, 50 + (income_expense_gap * 50) + (real_return_factor * 5)))
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Purchasing Power Loss", 
                    f"{purchasing_power_loss:.1f}%",
                    delta_color="inverse"
                )
                
                # Create inflation resilience gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=resilience_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Inflation Resilience Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF6B6B"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': resilience_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Income vs Expenses projection
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=income_projection,
                    mode='lines+markers',
                    name='Income',
                    line=dict(color='#43AA8B', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=expense_projection,
                    mode='lines+markers',
                    name='Expenses',
                    line=dict(color='#FF6B6B', width=3)
                ))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title="Income vs Expenses Projection",
                    xaxis_title="Years",
                    yaxis_title="Amount (₹)",
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Real returns on investments
            st.markdown("<h4>Real Returns on Investments (After Inflation)</h4>", unsafe_allow_html=True)
            real_returns_col1, real_returns_col2, real_returns_col3 = st.columns(3)

            with real_returns_col1:
                st.markdown(f"""
                <div class='metric-card' style='background-color: #333; color: white; padding: 10px; border-radius: 5px;'>
                    <div class='metric-value'>{fd_real_return:.1f}%</div>
                    <div class='metric-label' style='color: white;'>Fixed Deposits</div>
                </div>
                """, unsafe_allow_html=True)

            with real_returns_col2:
                st.markdown(f"""
                <div class='metric-card' style='background-color: #333; color: white; padding: 10px; border-radius: 5px;'>
                    <div class='metric-value'>{ppf_real_return:.1f}%</div>
                    <div class='metric-label' style='color: white;'>PPF/EPF</div>
                </div>
                """, unsafe_allow_html=True)

            with real_returns_col3:
                st.markdown(f"""
                <div class='metric-card' style='background-color: #333; color: white; padding: 10px; border-radius: 5px;'>
                    <div class='metric-value'>{equity_real_return:.1f}%</div>
                    <div class='metric-label' style='color: white;'>Equity</div>
                </div>
                """, unsafe_allow_html=True)

            
            # Financial suggestions
            st.markdown("<h4>Financial Suggestions</h4>", unsafe_allow_html=True)
            if resilience_score >= 80:
                st.markdown("""
                ✅ **Inflation Protected**: Your financial position shows strong resilience against inflation.
                
                **Recommendations:**
                - Continue investing in inflation-beating assets like equities and real estate
                - Consider Inflation-Indexed Bonds (IIBs) to protect part of your fixed income allocation
                - Maintain gold allocation of 5-10% as an inflation hedge
                - Negotiate annual salary increments that beat inflation by at least 2-3%
                - Consider REITs (Real Estate Investment Trusts) for passive real estate exposure
                """)
            elif resilience_score >= 50:
                st.markdown("""
                🟡 **Moderately Protected**: You have some protection against inflation but could be vulnerable.
                
                **Recommendations:**
                - Increase equity allocation to 50-60% of investments to beat inflation long-term
                - Consider Sovereign Gold Bonds for inflation protection with additional interest
                - Explore TIPS (Treasury Inflation-Protected Securities) if available
                - Reduce fixed deposit allocation and move to debt mutual funds with better returns
                - Invest in sectors likely to benefit from inflation (commodities, energy, consumer staples)
                - Focus on developing skills to increase earning potential above inflation rate
                """)
            else:
                st.markdown("""
                🔴 **Highly Vulnerable**: High inflation will significantly erode your purchasing power.
                
                **Urgent Recommendations:**
                - Immediately restructure investment portfolio to include inflation-beating assets
                - Reduce cash holdings to minimum required emergency fund
                - Invest in direct equities or equity mutual funds for long-term inflation protection
                - Consider investments in asset-heavy businesses which typically perform well in inflationary environments
                - Lock in fixed-rate loans before interest rates rise further
                - Look for government subsidy programs to reduce essential expenses
                - Develop additional income streams to offset rising costs
                - Consider indexed annuities for retirement planning
                """)
            st.markdown("</div>", unsafe_allow_html=True)

    # Financial Goals Assessment
    st.markdown("<h2 class='sub-header'>Financial Goals Assessment</h2>", unsafe_allow_html=True)

    # Goal selection
    goal_options = ["Home Purchase", "Car Purchase", "Higher Education", "Retirement", "Gold Investment"]
    selected_goal = st.selectbox("Select a financial goal to assess:", goal_options)

    # Initialize session state
    if 'property_value' not in st.session_state:
        st.session_state.property_value = 5000000

    if 'car_value' not in st.session_state:
        st.session_state.car_value = 800000

    # Collect user financial details
    st.sidebar.markdown("### Your Financial Details")
    monthly_income = st.sidebar.number_input("Monthly Income (₹)", min_value=10000, value=50000)
    monthly_savings = st.sidebar.number_input("Monthly Savings (₹)", min_value=1000, value=10000)
    debt_to_income_ratio = st.sidebar.slider("Debt-to-Income Ratio (%)", 0, 50, 20)

    if selected_goal == "Home Purchase":
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Home Purchase Goal</h3>", unsafe_allow_html=True)
            
            property_value = st.number_input(
                "Desired Property Value (₹)", 
                min_value=1000000,
                max_value=50000000,
                value=st.session_state.property_value,
                help="Enter the expected cost of the property"
            )
            st.session_state.property_value = property_value
            
            down_payment_percent = st.slider("Down Payment (%)", 10, 50, 20)
            loan_term_years = st.slider("Loan Term (Years)", 5, 30, 20)
            interest_rate = st.slider("Expected Loan Interest Rate (%)", 6.0, 12.0, 8.5, 0.1)
            time_horizon = st.slider("When do you plan to purchase (years)", 1, 15, 5)
            
            down_payment_amount = property_value * (down_payment_percent / 100)
            loan_amount = property_value - down_payment_amount
            
            # Calculate EMI
            monthly_interest_rate = interest_rate / 12 / 100
            loan_term_months = loan_term_years * 12
            emi = loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months / ((1 + monthly_interest_rate)**loan_term_months - 1)
            
            monthly_savings_needed = down_payment_amount / (time_horizon * 12)
            recommended_emi_limit = monthly_income * 0.4
            
            achievability_score = min(1, monthly_savings / monthly_savings_needed) * 100
            
        with col2:
            st.markdown("<h4>Loan Analysis</h4>", unsafe_allow_html=True)
            st.metric("Down Payment Required", f"₹{down_payment_amount:,.0f}")
            st.metric("Monthly EMI", f"₹{emi:,.0f}")
            st.metric("Loan Amount", f"₹{loan_amount:,.0f}")
            st.metric("Recommended Max EMI", f"₹{recommended_emi_limit:,.0f}")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=achievability_score,
                title={'text': "Goal Achievability Score"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#4ECDC4"}}
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'  
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    elif selected_goal == "Car Purchase":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Car Purchase Goal</h3>", unsafe_allow_html=True)
            car_value = st.number_input("Car Value (₹)", 300000, 10000000, st.session_state.car_value)
            st.session_state.car_value = car_value

            down_payment_percent = st.slider("Down Payment (%)", 10, 50, 20)
            loan_term_years = st.slider("Loan Term (Years)", 1, 7, 5)
            interest_rate = st.slider("Expected Loan Interest Rate (%)", 7.0, 15.0, 9.5, 0.1)
            time_horizon = st.slider("When do you plan to purchase (years)", 0, 5, 1)
            
            down_payment_amount = car_value * (down_payment_percent / 100)
            loan_amount = car_value - down_payment_amount
            
            monthly_interest_rate = interest_rate / 12 / 100
            loan_term_months = loan_term_years * 12
            emi = loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months / ((1 + monthly_interest_rate)**loan_term_months - 1)
                
        # Calculate EMI
        monthly_interest_rate = interest_rate / 12 / 100
        loan_term_months = loan_term_years * 12
        emi = loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months / ((1 + monthly_interest_rate)**loan_term_months - 1)
                
        # Monthly savings needed for down payment
        monthly_savings_needed = down_payment_amount / max(1, time_horizon * 12)
                
        # Affordability check
        recommended_emi_limit = total_monthly_income * 0.15  # Car EMI should not exceed 15% of income
        affordability_ratio = emi / recommended_emi_limit
                
        # Calculate achievability score
        down_payment_readiness = min(1, emergency_fund / down_payment_amount)
        emi_affordability = min(1, recommended_emi_limit / emi) if emi > 0 else 1
        total_cost_factor = min(1, 24 / (car_value / monthly_income))
        existing_debt_factor = max(0, min(1, (0.4 - debt_to_income_ratio / 100) / 0.4)) if debt_to_income_ratio > 0 else 1
        achievability_score = (down_payment_readiness * 0.3 + emi_affordability * 0.3 + total_cost_factor * 0.2 + existing_debt_factor * 0.2) * 100
            
        with col2:
                # Car loan breakdown
                st.markdown("<h4>Car Loan Analysis</h4>", unsafe_allow_html=True)
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Down Payment Required", f"₹{down_payment_amount:,.0f}")
                    st.metric("Monthly EMI", f"₹{emi:,.0f}")
                
                with metrics_col2:
                    st.metric("Loan Amount", f"₹{loan_amount:,.0f}")
                    ideal_emi_color = "normal" if emi <= recommended_emi_limit else "inverse"
                    st.metric("Recommended Max EMI", f"₹{recommended_emi_limit:,.0f}", f"{(emi - recommended_emi_limit):,.0f}", delta_color=ideal_emi_color)
                
                # Create achievability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=achievability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Goal Achievability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4ECDC4"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': achievability_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
        # Total cost of ownership analysis
        st.markdown("<h4>5-Year Cost of Ownership Analysis</h4>", unsafe_allow_html=True)
            
        # Calculate ownership costs
        fuel_cost_per_year = 50000  # Estimate
        insurance_per_year = car_value * 0.04  # 4% of car value
        maintenance_per_year = car_value * 0.02  # 2% of car value for first 5 years
        registration = car_value * 0.1  # One-time cost
        depreciation_5yr = car_value * 0.5  # 50% depreciation in 5 years
            
        # Ownership cost breakdown
        cost_categories = ['Loan Payments', 'Fuel', 'Insurance', 'Maintenance', 'Registration', 'Depreciation']
        cost_values = [
                emi * 12 * loan_term_years,
                fuel_cost_per_year * 5,
                insurance_per_year * 5,
                maintenance_per_year * 5,
                registration,
                depreciation_5yr
            ]
            
        cost_data = pd.DataFrame({
                'Category': cost_categories,
                'Amount': cost_values
            })
            
        fig = px.bar(
                cost_data,
                x='Category',
                y='Amount',
                title="5-Year Ownership Costs",
                color='Category',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
            
        # Total 5-year cost
        total_ownership_cost = sum(cost_values)
        monthly_ownership_cost = total_ownership_cost / (5 * 12)
            
        st.markdown(f"""
        <div style='background-color: #2D3142; padding: 1rem; border-radius: 8px;'>
            <div style='background-color: #2D3142; margin: 0;'>
                <h5 style='margin: 0; color: white;'>Total 5-Year Cost of Ownership: <span style='color: #ff6b6b;'>₹{total_ownership_cost:,.0f}</span></h5>
                <p style='margin-top: 0.5rem; color: white;'>Monthly Cost of Ownership: <span style='color: #ff6b6b;'>₹{monthly_ownership_cost:,.0f}</span> 
                ({(monthly_ownership_cost/total_monthly_income*100):.1f}% of your monthly income)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
            
        # Recommendations
        st.markdown("<h4>Action Plan & Recommendations</h4>", unsafe_allow_html=True)
        if achievability_score >= 80:
                st.markdown("""
                ✅ **Highly Achievable Goal**: This car purchase aligns well with your financial situation.
                
                **Action Plan:**
                1. Save ₹{:,.0f} monthly towards your down payment
                2. Compare car insurance quotes to get the best rates
                3. Consider increasing your down payment to 30% to reduce interest costs
                4. Research fuel efficiency ratings to reduce long-term ownership costs
                5. Consider extended warranty options for better protection
                6. Look into corporate/employee discounts with car manufacturers
                """.format(monthly_savings_needed))
        elif achievability_score >= 50:
                st.markdown("""
                🟡 **Moderately Achievable Goal**: You can buy this car with some financial adjustments.
                
                **Action Plan:**
                1. Consider a slightly less expensive model around ₹{:,.0f}
                2. Increase down payment to 30-40% to reduce monthly EMI burden
                3. Opt for a shorter loan tenure (3-4 years) to reduce total interest costs
                4. Research pre-owned certified vehicles for better value
                5. Consider delaying purchase by 6-12 months to build stronger down payment
                6. Look for year-end/festival season discounts to get better deals
                """.format(car_value * 0.8))
        else:
                st.markdown("""
                🔴 **Financially Strained Goal**: This car purchase may put significant strain on your finances.
                
                **Revised Action Plan:**
                1. Consider a more affordable car in the ₹{:,.0f} - ₹{:,.0f} range
                2. Explore pre-owned vehicles (2-3 years old) to get 30-40% lower price
                3. Delay purchase by 12-18 months to improve financial situation
                4. Increase down payment to at least 40% to significantly reduce EMI
                5. Consider car subscription or long-term rental services as alternatives
                6. Explore ride-sharing or carpooling options if applicable to your situation
                7. Use public transportation until your financial situation improves
                """.format(car_value * 0.5, car_value * 0.6))
                st.markdown("</div>", unsafe_allow_html=True)

    elif selected_goal == "Higher Education":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Higher Education Goal</h3>", unsafe_allow_html=True)
                
                course_fee = st.number_input("Total Course Fee (₹)", 100000, 5000000, 1200000)
                living_expenses = st.number_input("Annual Living Expenses (₹)", 0, 2000000, 300000)
                course_duration = st.slider("Course Duration (Years)", 1, 5, 2)
                time_horizon = st.slider("Years until education starts", 0, 15, 5)
                scholarship_amount = st.number_input("Expected Scholarship/Sponsorship (₹)", 0, 2000000, 0)
                
                # Calculate total education cost
                total_education_cost = course_fee + (living_expenses * course_duration) - scholarship_amount
                
                # Monthly savings needed
                monthly_savings_needed = total_education_cost / max(1, time_horizon * 12)
                
                # Education loan analysis
                loan_percent = st.slider("Percentage to be funded by loan", 0, 100, 60)
                education_loan_amount = total_education_cost * (loan_percent / 100)
                
                # Education loan EMI calculation
                loan_tenure_years = 7  # Standard education loan tenure
                education_loan_interest = 8.5  # Standard education loan interest rate
                monthly_interest_rate = education_loan_interest / 12 / 100
                loan_term_months = loan_tenure_years * 12
                expected_emi = education_loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**loan_term_months / ((1 + monthly_interest_rate)**loan_term_months - 1)
                
                # Affordability metrics
                savings_required = total_education_cost - education_loan_amount
                loan_affordability = min(1, total_monthly_income * 0.3 / expected_emi) if expected_emi > 0 else 1
                savings_affordability = min(1, monthly_savings / monthly_savings_needed) if monthly_savings_needed > 0 else 1
                existing_education_funds = 0
                fund_readiness = min(1, existing_education_funds / savings_required) if savings_required > 0 else 1
                
                # Calculate achievability score
                achievability_score = (loan_affordability * 0.4 + savings_affordability * 0.4 + fund_readiness * 0.2) * 100
            
            with col2:
                # Education cost breakdown
                st.markdown("<h4>Education Cost Analysis</h4>", unsafe_allow_html=True)
                
                # Create cost breakdown pie chart
                cost_labels = ['Course Fee', 'Living Expenses', 'Scholarship']
                cost_values = [course_fee, living_expenses * course_duration, -scholarship_amount]
                
                # Filter out zero or negative values
                filtered_labels = [label for label, value in zip(cost_labels, cost_values) if value > 0]
                filtered_values = [value for value in cost_values if value > 0]
                
                fig = px.pie(
                    names=filtered_labels,
                    values=filtered_values,
                    title="Education Cost Breakdown",
                    color_discrete_sequence=px.colors.sequential.Agsunset_r,
                    hole=0.4
                )
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Create achievability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=achievability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Goal Achievability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4ECDC4"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': achievability_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Funding analysis
            st.markdown("<h4>Education Funding Analysis</h4>", unsafe_allow_html=True)
            
            funding_col1, funding_col2, funding_col3 = st.columns(3)
            
            with funding_col1:
                st.markdown(f"""
                <div class='metric-card' style='background-color: #333; color: white; padding: 10px; border-radius: 5px;'>
                    <div class='metric-value'>₹{total_education_cost:,.0f}</div>
                    <div class='metric-label' style='color: white;'>Total Education Cost</div>
                </div>
                """, unsafe_allow_html=True)

            with funding_col2:
                st.markdown(f"""
                <div class='metric-card' style='background-color: #333; color: white; padding: 10px; border-radius: 5px;'>
                    <div class='metric-value'>₹{education_loan_amount:,.0f}</div>
                    <div class='metric-label' style='color: white;'>Expected Loan Amount</div>
                </div>
                """, unsafe_allow_html=True)

            with funding_col3:
                st.markdown(f"""
                <div class='metric-card' style='background-color: #333; color: white; padding: 10px; border-radius: 5px;'>
                    <div class='metric-value'>₹{expected_emi:,.0f}</div>
                    <div class='metric-label' style='color: white;'>Expected Monthly EMI</div>
                </div>
                """, unsafe_allow_html=True)

            
            # Savings projection
            if time_horizon > 0:
                st.markdown("<h4>Education Fund Savings Projection</h4>", unsafe_allow_html=True)
                
                # Create savings projection chart
                months = list(range(0, time_horizon * 12 + 1))
                savings_projection = [monthly_savings * m for m in months]
                target_line = [savings_required] * len(months)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=months,
                    y=savings_projection,
                    mode='lines',
                    name='Projected Savings',
                    line=dict(color='#4ECDC4', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=months,
                    y=target_line,
                    mode='lines',
                    name='Required Savings',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Months",
                    yaxis_title="Amount (₹)",
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("<h4>Action Plan & Recommendations</h4>", unsafe_allow_html=True)
            if achievability_score >= 80:
                st.markdown("""
                ✅ **Highly Achievable Goal**: This education goal is well within your financial reach.
                
                **Action Plan:**
                1. Start a dedicated education fund with SIP of ₹{:,.0f} per month
                2. Research and apply for multiple scholarships and grants
                3. Consider education-specific investment options like Sukanya Samriddhi Yojana (if applicable)
                4. Research tax-advantaged education savings vehicles available in India
                5. Prepare documentation for education loan pre-approval closer to admission
                6. Research part-time work opportunities during education to offset costs
                """.format(monthly_savings_needed))
            elif achievability_score >= 50:
                st.markdown("""
                🟡 **Moderately Achievable Goal**: This education goal requires careful planning.
                
                **Action Plan:**
                1. Increase monthly savings to ₹{:,.0f} by optimizing current expenses
                2. Research more affordable educational institutions or similar programs
                3. Consider education loans with better interest rates and terms
                4. Look for employer sponsorship or scholarship opportunities
                5. Consider part-time education options to continue earning
                6. Build a strong academic profile to qualify for merit scholarships
                """.format(monthly_savings_needed * 1.2))
            else:
                st.markdown("""
                🔴 **Challenging Goal**: This education goal may need significant adjustments.
                
                **Revised Action Plan:**
                1. Consider more affordable education options around ₹{:,.0f}
                2. Look into distance learning or online programs to reduce costs
                3. Research education loans with government subsidies
                4. Consider working for a few years to build savings
                5. Look for companies offering education assistance programs
                6. Explore similar programs in more affordable locations
                7. Consider breaking the goal into smaller phases
                """.format(total_education_cost * 0.6))
            st.markdown("</div>", unsafe_allow_html=True)

    elif selected_goal == "Retirement":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Retirement Planning</h3>", unsafe_allow_html=True)
                retirement_age = st.slider("Expected Retirement Age", age + 1, 80, 60)
                life_expectancy = st.slider("Life Expectancy", retirement_age + 1, 100, 85)
                expected_inflation = st.slider("Expected Inflation Rate (%)", 4.0, 10.0, 6.0, 0.1)
                expected_return = st.slider("Expected Investment Return (%)", 6.0, 15.0, 12.0, 0.1)
                
                # Calculate retirement metrics
                years_to_retirement = retirement_age - age
                retirement_duration = life_expectancy - retirement_age
                
                # Calculate required retirement corpus
                monthly_expenses_at_retirement = total_monthly_expenses * ((1 + expected_inflation/100) ** years_to_retirement)
                annual_expenses_at_retirement = monthly_expenses_at_retirement * 12
                required_corpus = annual_expenses_at_retirement * ((1 - (1 / ((1 + expected_inflation/100) ** retirement_duration))) / (expected_inflation/100))
                
                # Calculate required monthly savings
                monthly_investment_needed = (required_corpus * (expected_return/100/12)) / (((1 + expected_return/100/12) ** (years_to_retirement * 12)) - 1)
                
                # Calculate current retirement savings
                current_retirement_savings = ppf_epf + (stocks_value * 0.7) + (fd_value * 0.3)  # Assuming portions of investments for retirement
                
                # Calculate achievability metrics
                savings_ratio = min(1, monthly_savings / monthly_investment_needed) if monthly_investment_needed > 0 else 1
                corpus_ratio = min(1, current_retirement_savings / required_corpus)
                time_adequacy = min(1, years_to_retirement / 30)  # Assuming 30 years is ideal for retirement planning
                
                # Calculate achievability score
                achievability_score = (savings_ratio * 0.4 + corpus_ratio * 0.4 + time_adequacy * 0.2) * 100
            
            with col2:
                # Retirement corpus projection
                st.markdown("<h4>Retirement Corpus Analysis</h4>", unsafe_allow_html=True)
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Required Corpus", f"₹{required_corpus:,.0f}")
                    st.metric("Monthly Investment Needed", f"₹{monthly_investment_needed:,.0f}")
                
                with metrics_col2:
                    st.metric("Current Retirement Savings", f"₹{current_retirement_savings:,.0f}")
                    st.metric("Years to Retirement", f"{years_to_retirement}")
                
                # Create achievability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=achievability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Goal Achievability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4ECDC4"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': achievability_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Retirement corpus projection chart
            st.markdown("<h4>Retirement Corpus Projection</h4>", unsafe_allow_html=True)
            
            years = list(range(years_to_retirement + 1))
            corpus_projection = []
            current_corpus = current_retirement_savings
            
            for year in years:
                corpus_projection.append(current_corpus)
                current_corpus = current_corpus * (1 + expected_return/100) + (monthly_investment_needed * 12)
            
            target_line = [required_corpus] * len(years)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=years,
                y=corpus_projection,
                mode='lines',
                name='Projected Corpus',
                line=dict(color='#4ECDC4', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=target_line,
                mode='lines',
                name='Required Corpus',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Years",
                yaxis_title="Amount (₹)",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("<h4>Action Plan & Recommendations</h4>", unsafe_allow_html=True)
            if achievability_score >= 80:
                st.markdown("""
                ✅ **Strong Retirement Planning**: You're on track for a comfortable retirement.
                
                **Action Plan:**
                1. Continue monthly investment of ₹{:,.0f} towards retirement
                2. Maintain asset allocation of 60-70% equity for long-term growth
                3. Maximize EPF/PPF contributions for tax benefits
                4. Consider National Pension System (NPS) for additional tax benefits
                5. Review and rebalance portfolio annually
                6. Consider purchasing health insurance for retirement years
                """.format(monthly_investment_needed))
            elif achievability_score >= 50:
                st.markdown("""
                🟡 **Moderate Progress**: Your retirement planning needs some strengthening.
                
                **Action Plan:**
                1. Increase monthly retirement savings to ₹{:,.0f}
                2. Optimize investment portfolio for better returns
                3. Consider postponing retirement to age {:.0f} to build larger corpus
                4. Reduce current expenses to increase savings
                5. Explore additional income sources
                6. Start a separate health savings fund for retirement
                """.format(monthly_investment_needed * 1.2, retirement_age + 3))
            else:
                st.markdown("""
                🔴 **Significant Gap**: Your retirement planning needs immediate attention.
                
                **Urgent Action Plan:**
                1. Immediately start retirement savings of at least ₹{:,.0f} monthly
                2. Consider working longer - target retirement age of {:.0f}
                3. Significantly reduce current expenses
                4. Explore higher-paying career opportunities
                5. Consider post-retirement part-time work
                6. Review lifestyle expectations for retirement
                7. Start building multiple income streams
                """.format(monthly_investment_needed * 0.7, retirement_age + 5))
            st.markdown("</div>", unsafe_allow_html=True)

    elif selected_goal == "Gold Investment":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3>Gold Investment Planning</h3>", unsafe_allow_html=True)
                target_gold_value = st.number_input("Target Gold Investment Value (₹)", 50000, 10000000, 500000)
                investment_horizon = st.slider("Investment Horizon (Years)", 1, 20, 5)
                gold_return_rate = st.slider("Expected Gold Return Rate (%)", 5.0, 15.0, 8.0, 0.1)
                
                # Calculate required monthly investment
                monthly_investment_needed = (target_gold_value * (gold_return_rate/100/12)) / (((1 + gold_return_rate/100/12) ** (investment_horizon * 12)) - 1)
                
                # Calculate achievability metrics
                savings_ratio = min(1, monthly_savings / monthly_investment_needed) if monthly_investment_needed > 0 else 1
                current_gold_ratio = min(1, gold_value / target_gold_value)
                income_adequacy = min(1, (monthly_savings / monthly_income) / 0.2)  # Assuming 20% of income is ideal max for gold
                
                # Calculate achievability score
                achievability_score = (savings_ratio * 0.4 + current_gold_ratio * 0.3 + income_adequacy * 0.3) * 100
            
            with col2:
                # Gold investment analysis
                st.markdown("<h4>Gold Investment Analysis</h4>", unsafe_allow_html=True)
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Current Gold Value", f"₹{gold_value:,.0f}")
                    st.metric("Monthly Investment Needed", f"₹{monthly_investment_needed:,.0f}")
                
                with metrics_col2:
                    st.metric("Target Gold Value", f"₹{target_gold_value:,.0f}")
                    st.metric("Investment Horizon", f"{investment_horizon} years")
                
                # Create achievability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=achievability_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Goal Achievability Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4ECDC4"},
                        'steps': [
                            {'range': [0, 30], 'color': "#F94144"},
                            {'range': [30, 70], 'color': "#F9C74F"},
                            {'range': [70, 100], 'color': "#43AA8B"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': achievability_score
                        }
                    }
                ))
                
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Gold investment projection
            st.markdown("<h4>Gold Investment Projection</h4>", unsafe_allow_html=True)
            
            months = list(range(0, investment_horizon * 12 + 1))
            investment_projection = []
            current_value = gold_value
            
            for month in months:
                investment_projection.append(current_value)
                current_value = current_value * (1 + gold_return_rate/100/12) + monthly_investment_needed
            
            target_line = [target_gold_value] * len(months)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[m/12 for m in months],  # Convert months to years for x-axis
                y=investment_projection,
                mode='lines',
                name='Projected Value',
                line=dict(color='#4ECDC4', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=[m/12 for m in months],
                y=target_line,
                mode='lines',
                name='Target Value',
                line=dict(color='#FF6B6B', width=2, dash='dash')
            ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Years",
                yaxis_title="Amount (₹)",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Investment mode comparison
            st.markdown("<h4>Gold Investment Options Comparison</h4>", unsafe_allow_html=True)
            
            comparison_data = pd.DataFrame({
                'Feature': ['Liquidity', 'Safety', 'Returns', 'Storage Cost', 'Transaction Cost'],
                'Physical Gold': [3, 4, 3, 1, 2],
                'Gold ETF': [5, 4, 4, 5, 4],
                'Sovereign Gold Bonds': [2, 5, 5, 5, 5],
                'Digital Gold': [4, 4, 4, 5, 3]
            })
            
            fig = px.imshow(
                comparison_data.set_index('Feature'),
                color_continuous_scale='Teal',
                aspect='auto'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title='Investment Options Comparison (5 = Best, 1 = Worst)',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("<h4>Action Plan & Recommendations</h4>", unsafe_allow_html=True)
            if achievability_score >= 80:
                st.markdown("""
                ✅ **Well-Planned Gold Investment**: Your gold investment goal is realistic and achievable.
                
                **Action Plan:**
                1. Start monthly gold investment of ₹{:,.0f}
                2. Consider Sovereign Gold Bonds for better returns
                3. Diversify gold investments across physical and paper gold
                4. Set up automatic monthly investments
                5. Monitor gold prices for strategic buying
                6. Consider tax implications of different gold investment options
                """.format(monthly_investment_needed))
            elif achievability_score >= 50:
                st.markdown("""
                🟡 **Moderate Planning**: Your gold investment goal needs some adjustments.
                
                **Action Plan:**
                1. Start with smaller monthly investment of ₹{:,.0f}
                2. Extend investment horizon to {:.0f} years for better achievability
                3. Consider Gold ETFs for lower investment threshold
                4. Start SIP in Gold Funds
                5. Research tax-efficient gold investment options
                6. Monitor and rebalance overall portfolio allocation to gold
                """.format(monthly_investment_needed * 0.8, investment_horizon + 2))
            else:
                st.markdown("""
                🔴 **Needs Revision**: Your gold investment goal needs significant revision.
                
                **Revised Action Plan:**
                1. Reduce target gold investment to ₹{:,.0f}
                2. Start with minimum monthly investment of ₹{:,.0f}
                3. Consider digital gold for smaller investment amounts
                4. Extend timeline to {:.0f} years
                5. Focus on building emergency fund first
                6. Research gold accumulation plans
                7. Consider gold mutual funds for lower initial investment
                """.format(target_gold_value * 0.6, monthly_investment_needed * 0.5, investment_horizon + 3))
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅ Back to Dashboard"):
        st.session_state.menu_state = "main"
        st.rerun()