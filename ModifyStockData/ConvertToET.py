import pandas as pd
import pytz
import os
from datetime import time

def filter_market_hours(df, timestamp_col):
    # Define New York timezone
    ny_tz = pytz.timezone('America/New_York')
    
    # Convert timestamp column to datetime and localize to New York timezone
    df[timestamp_col] = pd.to_datetime(df[timestamp_col]).dt.tz_convert(ny_tz)
    
    # Define market start and end times
    market_start = time(9, 30)
    market_end = time(16, 0)
    
    # Filter rows where time is within market hours
    df = df[df[timestamp_col].dt.time.between(market_start, market_end)]
    
    # Return filtered DataFrame
    return df

def process_file(file_path):
    # Load the DataFrame
    df = pd.read_csv(file_path)
    
    # Assuming the timestamp column is named 'timestamp'
    df_filtered = filter_market_hours(df, 'timestamp')
    
    # Define the new file name
    new_file_name = os.path.basename(file_path)
    new_file_path = file_path  # Overwrite the original file, or modify this path as needed
    
    # Save the filtered DataFrame to a new file
    df_filtered.to_csv(new_file_path, index=False)
    print(f"Processed and saved: {new_file_path}")

def process_files_recursively(root_folder):
    # Walk through all directories and files in the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            process_file(file_path)

# Specify your root directory path here
root_folder = 'StockData/2022-01-01-2024-02-08/Individual'
process_files_recursively(root_folder)