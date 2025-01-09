import pandas as pd
import os

def calculate_percentage_change_for_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Ensure the 'close' column exists
    if 'close' in df.columns:
        # Calculate the percentage change in the 'close' value
        df['close_pct_change'] = df['close'].pct_change() * 100
        
        # Fill any NaN values with 0 (typically the first row)
        df['close_pct_change'].fillna(0, inplace=True)
        
        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
        print(f"Processed and updated file: {os.path.basename(file_path)}")
    else:
        print(f"'close' column not found in file: {os.path.basename(file_path)}")

def calculate_percentage_change(folder_path):
    # Walk through the directory and its subdirectories
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file_name in filenames:
            # Check if the file is a CSV
            if file_name.endswith('.csv'):
                file_path = os.path.join(dirpath, file_name)
                calculate_percentage_change_for_csv(file_path)
            else:
                print(f"Skipping non-CSV file: {file_name}")

# Replace 'your/folder/path' with the actual folder path containing your CSV files
folder_path = 'StockData/2022-01-01-2024-02-08/Individual'
calculate_percentage_change(folder_path)