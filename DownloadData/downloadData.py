import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pandas as pd
import os

# Alpaca API credentials
API_KEY = 'PKE0BV0FCFRYT1DSP494'
API_SECRET = 'VXmN4uPca1HrOXLpWfxR9MMsgIDql1aEK0F7xUJG'
BASE_URL = 'https://paper-api.alpaca.markets'  # or use https://api.alpaca.markets for live trading

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def save_stock_data(symbols, start_date=None, end_date=None, timeframe='1Min'):
    # Set default start and end dates if not provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Format directory name and create it if it doesn't exist
    directory = f"StockData/{start_date}-{end_date}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(os.path.join(directory, "Individual")):
        os.makedirs(os.path.join(directory, "Individual"))

    # Iterate over each symbol to fetch and save its data
    for symbol in symbols:
        if not os.path.exists(os.path.join(directory, "Individual", symbol)):
            os.makedirs(os.path.join(directory, "Individual", symbol))
        # Fetch historical data using the 'get_bars' method
        df = api.get_bars(symbol, timeframe, start=start_date, end=end_date, adjustment='raw').df
        # Check if the dataframe is not empty
        if not df.empty:
            # Define the file path and save the dataframe as a CSV
            file_path = os.path.join(directory, "Individual", symbol, f"{symbol}_{timeframe}.csv")
            df.to_csv(file_path)
            print(f"Data for {symbol} saved to {file_path}")
        else:
            print(f"No data found for {symbol}")

def load_symbols(file_path):
    """
    Load stock symbols from a given file.

    :param file_path: Path to the file containing stock symbols.
    :return: List of stock symbols.
    """
    with open(file_path, 'r') as file:
        symbols = file.read().splitlines()
    return symbols

# Load symbols from a file
file_path = 'Symbols/TechGiants'
symbols = load_symbols(file_path)

# Specify the date range and timeframe
start_date = '2022-01-01'  # Example start date
end_date = '2024-02-08'    # Example end date
#start_date = None
#end_date = None

timeframe = TimeFrame(1, TimeFrameUnit.Minute)         # Default to 1 minute

# Fetch and save stock data
save_stock_data(symbols, start_date, end_date, timeframe)