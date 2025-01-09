import pandas as pd
import os
import ta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands

# Function to calculate SMA, EMA, RSI, MACD, and Bollinger Bands
def calculate_indicators(df):
    df['SMA'] = ta.trend.sma_indicator(df['close'], window=14)
    df['EMA'] = ta.trend.ema_indicator(df['close'], window=14)
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    df['MACD'] = ta.trend.MACD(df['close'], window_slow=26, window_fast=12).macd()
    
    indicator_bb = BollingerBands(close=df['close'], window=20)
    df['BB_MA'] = indicator_bb.bollinger_mavg()
    df['BB_High'] = indicator_bb.bollinger_hband()
    df['BB_Low'] = indicator_bb.bollinger_lband()
    
    return df

# Function to process CSV files in a directory and its subdirectories
def process_csv_files(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv') and 'StdDev' not in filename:
                file_path = os.path.join(foldername, filename)
                df = pd.read_csv(file_path)
                
                # Check if the required columns are present
                required_columns = ['timestamp', 'close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap', 'close_pct_change']
                if all(col in df.columns for col in required_columns):
                    df = calculate_indicators(df)
                    
                    # Save the updated DataFrame to the same file
                    df.to_csv(file_path, index=False)
                    print(f'Processed: {file_path}')

# Specify the root directory where you want to start searching for CSV files
root_directory = 'StockData/2023-01-01-2024-02-08/Individual'

process_csv_files(root_directory)
