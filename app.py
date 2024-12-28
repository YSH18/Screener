import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from io import BytesIO

folder_id = '1pjW5_S83PMTUh0Kfy7XiyP69H0XJ1Hmt'

def authenticate_drive_api():
    # Access the 'installed' secret configuration
    installed_config = st.secrets.get("google", {})
    if not installed_config:
        st.error("Google installed credentials are missing or misconfigured in Streamlit secrets.")
    else:
        # Your existing code to authenticate
        flow = InstalledAppFlow.from_client_config(
            installed_config,
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
    st.write(st.secrets.keys())
    creds = None

    # Check if there are already saved credentials in the session
    if 'credentials' in st.session_state:
        creds = st.session_state['credentials']

    # If no valid credentials, go through the login process
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # The authentication step to obtain new credentials
            creds = flow.run_local_server(port=0)  # This will open a browser for authentication

        # Save credentials to session for future use
        st.session_state['credentials'] = creds

    # Build the API client
    return build("drive", "v3", credentials=creds)

# Call the authenticate function to initialize the API client
drive_service = authenticate_drive_api()


def list_files_in_folder(service, folder_id):
    """Lists all files in a specified Google Drive folder, handling pagination."""
    try:
        files = []
        page_token = None

        while True:
            # Create the query to list the files in the specified folder
            query = f"'{folder_id}' in parents"

            # Call the Drive API to list the files in the folder
            response = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name)",
                                            pageToken=page_token).execute()

            # Add the files from the current page to the list
            files.extend(response.get('files', []))

            # Check if there is a nextPageToken, which means there are more files to fetch
            page_token = response.get('nextPageToken')
            if not page_token:
                break  # No more pages, stop the loop

        return files

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def download_file(service, file_id):
    """Download a file from Google Drive and return its content as a DataFrame."""
    try:
        print(f"Starting download of file with ID: {file_id}")

        # Request to download the file
        request = service.files().get_media(fileId=file_id)
        file_content = BytesIO()  # Create an in-memory file buffer
        downloader = MediaIoBaseDownload(file_content, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()  # Progress indicator
            print(f"Download progress: {int(status.progress() * 100)}%")

        # After download, check the size of the file
        print(f"Download completed. File size: {file_content.getbuffer().nbytes} bytes.")

        # Ensure we are correctly reading the file content into a DataFrame
        file_content.seek(0)  # Reset the file pointer to the beginning after download
        try:
            print("Attempting to read the file into a DataFrame...")
            df = pd.read_excel(file_content, engine='openpyxl')
            print("File read successfully into DataFrame.")
            return df
        except Exception as e:
            print(f"Error reading the Excel file: {e}")
            return None
    except HttpError as error:
        print(f"Google Drive API Error: {error}")
        return None
    except Exception as e:
        print(f"General error occurred during file download or processing: {e}")
        return None


def load_stock_data(stock_symbol, folder_id):
    service = authenticate_drive_api()
    files = list_files_in_folder(service, folder_id)
    print(f"Found {len(files)} files in the folder.")
    print("Files in folder:", [file['name'] for file in files])
    # Log the stock symbol to see if it's passed correctly
    print(f"Looking for file: {stock_symbol}_div.xlsx")

    if not files:
        return None

    file_to_download = next((file for file in files if file['name'].lower() == f"{stock_symbol.lower()}_div.xlsx"),
                            None)

    if not file_to_download:
        st.error(f"Data file for stock symbol '{stock_symbol}' not found.")
        return None
    return download_file(service, file_to_download['id'])


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

def check_indicators_and_save(df, min_volume, min_price, min_banker_value):
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
            mask &= (df['volume'] >= min_volume * 100000)
        if min_price:
            mask &= (df['close'] >= min_price)
        if min_banker_value:
            mask &= (df['brsi'] >= min_banker_value)  # Assuming `brsi` column exists

        # Get matching symbols
        matching_symbols = [str(symbol) for symbol in df[mask]['symbol'].tolist()]

        # Create a temporary file to save the filtered results
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
            # Write the matching symbols to the file
            temp_file.write("\n".join(matching_symbols))
            temp_file_path = temp_file.name  # Get the path for the saved file

        # Return the matching symbols and file path for further use (could be used to read back or process further)
        return matching_symbols, temp_file_path

    except Exception as e:
        # Add more detailed error info
        st.error(f"Error processing data: {str(e)}, Data type of the DataFrame: {type(df)}")
        return [], None


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
    df = load_stock_data(stock_symbol,'1pjW5_S83PMTUh0Kfy7XiyP69H0XJ1Hmt')
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
    temp_file_path = None

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

            with col5:
                if st.button("OK"):
                    st.session_state['show_list'] = False
                    try:
                        # Authenticate and download the file content (this returns a DataFrame directly)
                        service = authenticate_drive_api()
                        file_id = '11WYYqM1vyVaRrAvnu5Wkef8WHMiofD0O'  # meeting criteria file
                        file_content = download_file(service, file_id)

                        # Print the type of file_content to check if it's already a DataFrame
                        print(
                            f"Downloaded file content type: {type(file_content)}")  # Should print <class 'pandas.core.frame.DataFrame'>

                        # Ensure that the file_content is a DataFrame
                        if isinstance(file_content, pd.DataFrame):
                            df = file_content  # Directly use it as the DataFrame
                        else:
                            st.error("Error: The downloaded file content is not a DataFrame.")
                            return  # Exit the function early

                        if df.empty:
                            st.error("DataFrame is empty!")
                            return  # Exit the function early

                        # Proceed with filtering the data and saving the results
                        matching_symbols, temp_file_path = check_indicators_and_save(df, min_volume, min_price,
                                                                                     min_banker_value)

                        if temp_file_path is None:
                            st.warning("No path found or file not saved.")
                            return  # Exit the function early

                        # Display the results of the filtering
                        if matching_symbols:
                            st.success("Matching stocks are found!")
                            st.session_state['show_list'] = True
                        else:
                            st.warning("No stocks found matching all selected criteria.")

                    except Exception as e:
                        # Catch any other errors that occur during the process
                        st.error(f"Error processing data at button: {str(e)}")

                # Display results
                if st.session_state['show_list']:
                    if temp_file_path:
                        try:
                            with open(temp_file_path, "r") as file:
                                stock_list = file.read().splitlines()
                            if stock_list:
                                num_matching_stocks = len(matching_symbols)
                                st.write(f"Number of stocks meeting the criteria: {num_matching_stocks}")
                                display_symbols_in_columns(stock_list)  # Call the function to display symbols in columns
                            else:
                                st.warning("No matching stocks found in the file.")
                        except Exception as e:
                            st.error(f"Error reading results file: {str(e)}")

            # Download button
            with col6:
                if temp_file_path:
                    try:
                        with open(temp_file_path, "r") as file:
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
