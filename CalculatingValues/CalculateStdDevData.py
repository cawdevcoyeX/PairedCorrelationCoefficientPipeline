import os
import pandas as pd
from scipy.stats import norm

def calculate_bounds(df, column, percentage):
    """
    Calculate the upper and lower bounds for the specified percentage.
    Args:
    - df: Pandas DataFrame containing the data.
    - column: The column name for which to calculate the bounds.
    - percentage: The percentage for which to calculate the bounds.

    Returns:
    - A tuple containing the lower and upper bounds.
    """
    mean = df[column].mean()
    std = df[column].std()
    z_score = norm.ppf(1 - ((1 - (percentage / 100)) / 2))
    lower_bound = mean - (z_score * std)
    upper_bound = mean + (z_score * std)
    return lower_bound, upper_bound

def process_csv_file(file_path, percentages):
    """
    Process a single CSV file to calculate and output bounds.
    Args:
    - file_path: The path to the CSV file.
    - percentages: A list of percentages for which to calculate bounds.
    """
    df = pd.read_csv(file_path)
    if 'close' in df.columns:
        results = {}
        for percentage in percentages:
            lower_bound, upper_bound = calculate_bounds(df, 'close_pct_change', percentage)
            results[f'{percentage}%'] = {'Lower Bound': lower_bound, 'Upper Bound': upper_bound}
        
        # Convert results to a DataFrame and write to a CSV file
        output_df = pd.DataFrame.from_dict(results, orient='index')
        output_path = f"{os.path.splitext(file_path)[0]}_StdDev.csv"
        output_df.to_csv(output_path)
        print(f"Output written to {output_path}")

def process_directory(root_dir, percentages=[90, 95, 99]):
    """
    Walk through the directory structure, starting from root_dir, and process each CSV file found.
    Args:
    - root_dir: The root directory to start the search from.
    - percentages: A list of percentages for which to calculate bounds.
    """
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and 'StdDev' not in file:
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {file_path}")
                process_csv_file(file_path, percentages)

# Specify the root directory to start from
root_dir = 'StockData/2022-01-01-2024-02-08/Individual'
percentages=[90, 95, 99]
# Call the function to start processing
process_directory(root_dir, percentages)