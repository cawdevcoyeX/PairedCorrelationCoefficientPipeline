import os
import pandas as pd

def rename_index_column_to_timestamp(csv_file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Check if the 'index' column exists
        if 'index' in df.columns:
            # Rename the 'index' column to 'timestamp'
            df.rename(columns={'index': 'timestamp'}, inplace=True)

            # Save the modified DataFrame back to the CSV file, overwriting the original
            df.to_csv(csv_file_path, index=False)

            print(f"Renamed 'index' column to 'timestamp' in {csv_file_path}")

    except Exception as e:
        print(f"Error processing {csv_file_path}: {str(e)}")

def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                rename_index_column_to_timestamp(csv_file_path)

if __name__ == "__main__":
    # Specify the starting folder
    starting_folder = 'StockData/2022-01-01-2024-02-08/Individual'

    # Start processing the folder and its subfolders
    process_folder(starting_folder)
