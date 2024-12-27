import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Define user credentials
USER_CREDENTIALS = {
    "user1": "123",
    "user2": "456",
    "admin": "admin123"
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.session_state['page'] = "Stock Screener"
    st.session_state['selected_stock'] = None
    st.session_state['show_list'] = False


# Function to handle login
def login(username, password):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['page'] = 'Stock Screener'
        st.success(f"Welcome, {username}!")
    else:
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.session_state['page'] = ''
        st.error("Invalid username or password, please try again.")


# Function to handle logout
def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = ""
    st.session_state['page'] = "Stock Screener"
    st.session_state['selected_stock'] = None
    st.session_state['show_list'] = False
    st.info("You have been logged out.")

def display_symbols_in_columns(symbols):
    """
    Display a list of stock symbols in 3 columns and trigger stock selection on button click.

    Args:
        symbols (list): List of stock symbols to display.
    """
    # Number of columns
    num_columns = 2
    # Calculate how many symbols should go in each column
    columns = st.columns([1,1])

    # Distribute the symbols across the columns
    symbols_per_column = len(symbols) // num_columns
    remaining = len(symbols) % num_columns

    start_index = 0
    for i in range(num_columns):
        end_index = start_index + symbols_per_column + (1 if i < remaining else 0)
        with columns[i]:
            # Iterate through each stock symbol and display it as a button
            for stock in symbols[start_index:end_index]:
                stock = stock.strip()
                if stock:  # Ensure it's not an empty string
                    # Use an on_click for better performance
                    if st.button(stock, key=f"btn_{stock}", on_click=select_stock, args=(stock,)):
                        pass
                        # Call select_stock only if a button is clicked
                        st.session_state['page'] = "Chart Viewer"

        start_index = end_index


# Function to handle stock selection
def select_stock(stock):
    st.session_state['selected_stock'] = stock
    st.session_state['page'] = "Chart Viewer"


# Function to check selected indicators and filter the stocks
def check_indicators_and_save(df, min_volume, min_price, min_banker_value):
    """
    Filter stocks based on selected checkboxes and save matching symbols to a .txt file.

    Args:
        df (pd.DataFrame): DataFrame containing the screener results.

    Returns:
        list: Symbols that meet all selected criteria.
    """
    try:
        # Initialize mask for filtering
        mask = pd.Series([True] * len(df))

        # Apply filters based on checkbox states
        if st.session_state.get('rainbow_check', False):
            mask &= (df['rainbow'] == 1)
        if st.session_state.get('y1_check', False):
            mask &= (df['y1'] == 1)
        if st.session_state.get('zj_check', False):
            mask &= (df['zj'] == 1)
        if st.session_state.get('qs_check', False):
            mask &= (df['qs'] == 1)

        # Apply user-defined filters
        if min_volume:
            mask &= (df['volume'] >= min_volume*100000)
        if min_price:
            mask &= (df['close'] >= min_price)
        if min_banker_value:
            mask &= (df['brsi'] >= min_banker_value)  # Assuming `banker_value` column exists

        # Get matching symbols
        matching_symbols = df[mask]['symbol'].tolist()

        # Save results to a .txt file
        output_path = "C:/Users/Cynthia Yeoh/Desktop/screener/test/filtered_results.txt"
        with open(output_path, "w") as file:
            file.write("\n".join(matching_symbols))

        return matching_symbols

    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return []


# Function to load stock data
def load_stock_data(stock_symbol):
    file_path = f"C:/Users/Cynthia Yeoh/Desktop/screener/TV historical data dividend (20241023)/{stock_symbol}_div.xlsx"
    try:
        df = pd.read_excel(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except FileNotFoundError:
        st.error(f"Data file for stock symbol '{stock_symbol}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Function to calculate fund flow
def calculate_fund_flow(df):
    close_prices = df['close']

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def sma(series, span):
        return series.rolling(window=span).mean()

    fast_ma = ema(close_prices, 4)
    slow_ma = ema(close_prices, 20)
    macd = fast_ma - slow_ma
    signal = ema(macd, 21)

    fast_ma_c = ema(macd, 3)
    slow_ma_c = sma(macd, 6)

    higher = np.zeros(len(df))
    lower = np.zeros(len(df))
    palette = []

    for i in range(1, len(df)):
        if fast_ma_c.iloc[i] <= slow_ma_c.iloc[i] and fast_ma_c.iloc[i - 1] >= slow_ma_c.iloc[i - 1]:
            higher[i] = max(fast_ma_c.iloc[i], fast_ma_c.iloc[i - 1])
            lower[i] = min(fast_ma_c.iloc[i - 1], slow_ma_c.iloc[i])
        elif fast_ma_c.iloc[i] >= slow_ma_c.iloc[i] and fast_ma_c.iloc[i - 1] <= slow_ma_c.iloc[i - 1]:
            higher[i] = max(fast_ma_c.iloc[i], slow_ma_c.iloc[i])
            lower[i] = min(slow_ma_c.iloc[i - 1], slow_ma_c.iloc[i])
        else:
            higher[i] = max(fast_ma_c.iloc[i], slow_ma_c.iloc[i - 1])
            lower[i] = min(fast_ma_c.iloc[i - 1], slow_ma_c.iloc[i])

        palette.append('red' if fast_ma_c.iloc[i] >= slow_ma_c.iloc[i] else 'lime')

    if len(palette) < len(df):
        palette.insert(0, 'lime')

    return higher, lower, palette, signal, slow_ma_c


# Function to display chart
def display_chart(stock_symbol):
    df = load_stock_data(stock_symbol)
    if df is not None:
        higher, lower, palette, signal, slow_ma_c = calculate_fund_flow(df)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.2, 0.1],
            vertical_spacing=0.03,
        )

        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Add volume trace
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['volume'],
                name="Volume",
                marker_color='blue',
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        # Add Fund Flow indicator traces
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=slow_ma_c,
                line=dict(color='rgba(255, 255, 255, 0)'),
                name="Slow MA",
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
                y=signal,
                fill='tonexty',
                fillcolor='lightskyblue',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name="Signal",
            ),
            row=3,
            col=1,
        )

        for i in range(len(df)):
            fig.add_trace(
                go.Scatter(
                    x=[df['datetime'].iloc[i], df['datetime'].iloc[i]],
                    y=[lower[i], higher[i]],
                    mode='lines',
                    line=dict(color=palette[i], width=2),
                    showlegend=False
                ),
                row=3,
                col=1,
            )

        fig.update_layout(
            title=f"Candlestick Chart {stock_symbol}",
            xaxis=dict(title=None, showgrid=False),
            xaxis3=dict(title="日期", showgrid=False),
            yaxis=dict(title="股价", side="left"),
            yaxis2=dict(title="交易量", side="left"),
            yaxis3=dict(title="资金所向", side="left"),
            height=800,
            showlegend=False,
            xaxis_rangeslider_visible=False,
        )

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        st.plotly_chart(fig, use_container_width=True)


# Main function
def main():
    st.title("选股平台")
    # Initialize the number of matching stocks
    num_matching_stocks = 0

    if not st.session_state['logged_in']:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(username, password)
            if st.session_state['logged_in']:
                st.rerun()

    if st.session_state['logged_in']:
        st.sidebar.title(f"Welcome, {st.session_state['username']}!")
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()

        # Navigation
        st.sidebar.subheader("Navigation")
        current_page = st.sidebar.radio("Choose Page", ["Stock Screener", "Chart Viewer"],
                                        key='navigation',
                                        index=0 if st.session_state['page'] == "Stock Screener" else 1)
        st.session_state['page'] = current_page

        if current_page == "Stock Screener":
            st.subheader("Stock Screener")

            st.session_state['show_list'] = False

            # Replace selectbox with checkboxes
            st.write("Select indicators (stocks must meet all selected criteria):")
            col1, col2 = st.columns(2)

            with col1:
                rainbow_selected = st.checkbox("彩图", key="rainbow_check")
                y1_selected = st.checkbox("第一黄柱", key="y1_check")

            with col2:
                zj_selected = st.checkbox("资金所向", key="zj_check")
                qs_selected = st.checkbox("趋势专家", key="qs_check")

            # Count selected indicators
            selected_count = sum([
                st.session_state.get('rainbow_check', False),
                st.session_state.get('y1_check', False),
                st.session_state.get('zj_check', False),
                st.session_state.get('qs_check', False)
            ])

            st.write("Filter by user-defined values (optional):")

            # Minimum Volume Input
            min_volume = st.number_input("Minimum Volume as a multiple of 100,000", min_value=0, value=0, step=1,
                                         help="Example 5 means 500,000")

            # Minimum Stock Price Input
            min_price = st.number_input("Minimum Stock Price", min_value=0.0, value=0.0, step=0.1,
                                        help="Enter 0.3 means will screen stock price with 0.3 and above")

            # Minimum Banker Value Input
            min_banker_value = st.number_input("Minimum Banker Value (0-100) ", min_value=0, value=0, step=1,
                                               help="Enter 30 means will screen banker value with 30 and above")

            col5, col6 = st.columns([1, 1])

            # Show list button
            with col5:
                if st.button("OK") :
                    st.session_state['show_list'] = False
                    # Read the Excel file
                    try:
                        file_path = r"C:\Users\Cynthia Yeoh\Desktop\screener\screened result\screened_nodered_20241224.xlsx"
                        df = pd.read_excel(file_path)

                        # Get matching symbols
                        matching_symbols = check_indicators_and_save(df, min_volume,min_price,min_banker_value)

                        if matching_symbols:
                            st.success("Matching stocks are found!")
                            num_matching_stocks = len(matching_symbols)
                            st.session_state['show_list'] = True
                        else:
                            st.warning("No stocks found matching all selected criteria.")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")

                # Display results
                if st.session_state['show_list']:
                    file_path = "C:/Users/Cynthia Yeoh/Desktop/screener/test/filtered_results.txt"
                    try:
                        with open(file_path, "r") as file:
                            stock_list = file.read().splitlines()
                        if stock_list:
                            st.write(f"Number of stocks meeting the criteria: {num_matching_stocks}")
                            display_symbols_in_columns(stock_list)  # Call the function to display symbols in columns
                        else:
                            st.warning("No matching stocks found in the file.")
                    except Exception as e:
                        st.error(f"Error reading results file: {str(e)}")

            # Download button
            with col6:
                try:
                    file_path = "C:/Users/Cynthia Yeoh/Desktop/screener/test/filtered_results.txt"
                    with open(file_path, "r") as file:
                        file_data = file.read()
                    if file_data:
                        st.download_button(
                            label="Download File",
                            data=file_data,
                            file_name="filtered_results.txt",
                            mime="text/plain"
                        )
                except FileNotFoundError:
                    st.warning("No file available to download.")

            # Allow manual input of stock symbol
            stock_input = st.text_input("Or enter a stock symbol for chart viewing:")

            # Submit button for manual input
            submit_button = st.button("Submit Stock")

            # Handle stock symbol input and page navigation

            if submit_button and stock_input:
                # Remove 'MYX:' prefix if present for file loading
                display_stock = stock_input.replace('MYX:', '')
                select_stock(display_stock)
                # Debugging: Check if session state is updated
                # st.write(f"Page set to: {st.session_state.get('page', 'No page set')}")
                st.session_state['page'] = "Chart Viewer"
                # Rerun to navigate to Chart Viewer #need to press enter and then submit button
                st.rerun()

        elif current_page == "Chart Viewer":
            st.subheader("Chart Viewer")
            # If there's a selected stock from the screener or entered manually, use it
            if st.session_state.get('selected_stock'):
                # Remove 'MYX:' prefix if present for file loading
                display_stock = st.session_state['selected_stock'].replace('MYX:', '')
                display_chart(display_stock)

if __name__ == "__main__":
    main()
