import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

def upload_data():
    """Upload Data page with data preview and editing capabilities"""

    # CSS
    st.markdown("""
    <style>
    * {
        font-family: Verdana, sans-serif !important;
    }
    /* Custom styling for Streamlit elements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        width: 100px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        background-color: #585c5c !important;
        gap: 1px;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0e7d8b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.write("## 📤 Upload Your Daily Transactions")
    
    # Initialize session state to store the dataframe
    if 'finance_df' not in st.session_state:
        st.session_state.finance_df = None
    
    # File uploader
    uploaded_file = st.file_uploader("Give us your existing financial history data to get started with the financial insights and explore the application features. Alternatively, you can navigate to the Expense or Income page as well, to manually enter transactions from scratch or add new ones as you continue using the app.", type="csv")
        
    # Process the uploaded file
    if uploaded_file is not None:
        # Only read the file if we haven't already or if a new file is uploaded
        if st.session_state.finance_df is None or uploaded_file.name != st.session_state.get('last_uploaded_file', ''):
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Store in session state
                st.session_state.finance_df = df
                st.session_state.last_uploaded_file = uploaded_file.name
                
                st.success(f"Successfully loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")
                return
    
    # If we have data, display and allow editing
    if st.session_state.finance_df is not None:
        df = st.session_state.finance_df

        # Option to clear the entire dataset
        if st.button("❌ Clear Dataset", help="This will remove the entire dataset"):
            st.session_state.finance_df = None
            st.session_state.pop('last_uploaded_file', None)
            st.success("Dataset cleared successfully. You can now upload a new file.")
            st.rerun()
        
        # Show basic statistics
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            date_cols = [col for col in df.columns if 'date' in col.lower()]

            if date_cols:
                date_col = date_cols[0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], format='%d-%m-%Y').dt.date 
                    min_date = df[date_col].min().strftime('%b %Y') if not pd.isna(df[date_col].min()) else "N/A"
                    max_date = df[date_col].max().strftime('%b %Y') if not pd.isna(df[date_col].max()) else "N/A"
                    st.metric("Date Range", f"{min_date} to {max_date}")
                except Exception as e:
                    st.warning(f"Failed to convert '{date_col}' to date format. Error: {e}")
        
        # Display tabs for viewing and editing
        view_tab, edit_tab = st.tabs(["View", "Edit"])
        
        with view_tab:
            st.write("#### Data Preview")
            st.dataframe(df)
            
            # Option to download the current dataframe
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Current Data",
                data=csv,
                file_name=f"finance_data.csv",
                mime="text/csv"
            )
        
        with edit_tab:
            st.write("#### Edit Your Data")
            st.write("To record fresh transactions, please go to Income/Expense Section!")
            st.write("To edit any of the existing cell values, directly make changes on the screen.\
            Select one or more columns to remove them using the 'Delete Rows' button at the bottom.\
            Don't forget to save your changes!")
            
            # Create an editable grid
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
            gb.configure_selection('multiple', use_checkbox=True)
            gb.configure_default_column(editable=True, groupable=True)
            
            # Make date columns use date editors if they exist
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            for date_col in date_cols:
                gb.configure_column(date_col, type=["dateColumnFilter", "customDateTimeFormat"], 
                                  custom_format_string='dd-MM-yy')
            
            # Make amount columns use number editors if they exist
            amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'price', 'cost', 'value'])]
            for amount_col in amount_cols:
                gb.configure_column(amount_col, type=["numericColumn", "numberColumnFilter", "customNumericFormat"], 
                                  precision=2)
            
            grid_options = gb.build()
            
            # Display the grid
            grid_return = AgGrid(df, 
                               gridOptions=grid_options,
                               update_mode=GridUpdateMode.MODEL_CHANGED,
                               fit_columns_on_grid_load=True,
                               height=400,
                               allow_unsafe_jscode=True)
            
            # Get the updated dataframe
            updated_df = grid_return['data']

            # Convert selected_rows to a DataFrame
            selected_rows = pd.DataFrame(grid_return['selected_rows'])

            # Delete selected rows
            if not selected_rows.empty and st.button('Delete Rows'):
                st.session_state.finance_df = updated_df[~updated_df.isin(selected_rows).all(axis=1)].reset_index(drop=True)
                st.success(f"Deleted {len(selected_rows)} rows")
                st.rerun()
            
            # Save button
            if st.button('Save Changes'):
                # Update session state with the edited dataframe
                st.session_state.finance_df = updated_df
                st.success("Changes synced successfully!")
                
    else:
        # Show placeholder when no file is uploaded
        st.info("Please upload a CSV file to get started.")
        
        # Option to use sample data
        if st.button("Use Sample Data"):
            # Create sample finance data
            sample_data = {
                'Date': [
                    '12-04-2022','13-04-2022','13-04-2022','14-04-2022','15-04-2022',
                    '15-05-2022','15-05-2022','16-05-2022','16-05-2022','17-05-2022'
                ],
                'Mode': [
                    'Credit Card','Credit Card', 'Bank Transfer', 'UPI', 'Debit Card', 
                    'Credit Card', 'UPI' , 'Mobile Wallet', 'Cash', 'Net Banking'
                ],
                'Category': [
                    'Groceries', 'Utilities', 'Rent', 'Communication', 'Education',
                    'Shopping', 'Care Essentials', 'Entertainment', 'Domestic Help', 'Income'
                ],
                'Remark': [
                    'supermarket', 'electricity bill', 'april rent paid', 'wifi bill', 'tuition fees',
                    'designer wear', 'face cream', 'movie tickets', 'maid salary', 'april salary credited'
                ],
                'Amount': [
                    2000, 1000, 15000, 950, 10000, 2800, 100, 570, 700, 50000
                ],
                'Income_Expense': [
                    'Expense','Expense','Expense','Expense','Expense',
                    'Expense','Expense','Expense','Expense','Income'
                ],
                'Transaction_id': [
                    'TXN1','TXN2','TXN3','TXN4','TXN5','TXN6','TXN7','TXN8','TXN9','TXN10'
                ]
            }
            
            sample_df = pd.DataFrame(sample_data)
            st.session_state.finance_df = sample_df
            st.session_state.last_uploaded_file = "sample_data.csv"
            st.rerun()