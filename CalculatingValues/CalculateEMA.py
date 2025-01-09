import os
import pandas as pd
from glob import glob

def find_csv_files(root_dir, pattern="*.csv"):
    """Recursively find all CSV files in the root_dir that do not contain 'StdDev' in their name."""
    files = [y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], pattern)) if 'StdDev' not in y]
    return files

def load_process_and_save_csv(files, gamma_values):
    """Load CSV files, focus on the 'close' column, apply smoothing, calculate percent changes, fill NaNs, and save."""
    for file in files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for gamma in gamma_values:
            ema_column = f'EMA_{gamma}'
            pct_change_column = f'PctChange_{gamma}'
            
            df[ema_column] = df['close'].ewm(span=gamma, adjust=False).mean()  # Calculate EMA
            df[pct_change_column] = df[ema_column].pct_change().fillna(0) * 100  # Calculate % change and fill NaNs with 0
            
        df.to_csv(file, index=False)  # Save the DataFrame back to the same CSV file
        print("Saved: ",file)

# Example usage
root_dir = 'StockData/2022-01-01-2024-02-08/Individual'  # Replace with your folder path
gamma_values = [1, 2, 3]  # List of gamma values to use for smoothing

csv_files = find_csv_files(root_dir)
load_process_and_save_csv(csv_files, gamma_values)
