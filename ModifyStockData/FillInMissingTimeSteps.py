import os
import pandas as pd
from datetime import datetime, timedelta

# Function to fill missing timestamps in a dataframe within the specified time range
def fill_missing_timestamps(df, timestep):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    
    new_time_range = pd.date_range(start=start_time, end=end_time, freq=timestep)
    
    df = df.set_index('timestamp')
    df = df.reindex(new_time_range)
    
    # Filter rows within the time range of 9 am to 4 pm (Monday to Friday)
    df = df.between_time("09:30:00", "16:00:00")
    
    df = df.reset_index()  # Reset the index to make "timestamp" a regular column
    
    return df

# Function to process a CSV file
def process_csv_file(file_path, timestep):
    try:
        df = pd.read_csv(file_path)
        df = fill_missing_timestamps(df, timestep)
        
        # Extract the original file name without extension
        file_name, file_extension = os.path.splitext(os.path.basename(file_path))
        
        # Create the new file name
        updated_file_name = os.path.join(os.path.dirname(file_path), f"{file_name}.csv") #_updated.csv")
        
        # Save the updated dataframe to the new CSV file
        df.to_csv(updated_file_name, index=False)  # Do not save index as an extra column
        print(f"Processed: {file_path} and saved as {updated_file_name}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# Function to recursively process CSV files in a folder and its subfolders
def process_folder(root_folder, timestep):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                process_csv_file(file_path, timestep)

if __name__ == "__main__":
    folder_to_process = "StockData/2022-01-01-2024-02-08/Individual"  # Replace with the path to your folder
    timestep = "1T"  # 5-minute intervals

    process_folder(folder_to_process, timestep)
