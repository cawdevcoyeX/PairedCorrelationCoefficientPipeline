import os
import pandas as pd

def cast_to_step(value, step, lower_bound, upper_bound):
    # Check if value is NaN and return it as is if true
    if pd.isna(value):
        return value
    
    step = (upper_bound - lower_bound) / step

    value = min(value, upper_bound)
    # Ensure value falls within the bounds before casting to step
    value = max(min(value, upper_bound), lower_bound)
    # Find the nearest step value and return it
    return round((value - lower_bound) / step) * step + lower_bound

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_files(base_folder, confidence, granularity):
    individual_folder = os.path.join(base_folder, 'Individual')
    for stock_folder in os.listdir(individual_folder):
        stock_folder_path = os.path.join(individual_folder, stock_folder)
        if os.path.isdir(stock_folder_path):
            for file in os.listdir(stock_folder_path):
                if not 'StdDev' in file and file.endswith('.csv'):
                    data_file_path = os.path.join(stock_folder_path, file)
                    std_dev_file_path = os.path.join(stock_folder_path, file.replace('.csv', '_StdDev.csv'))
                    
                    # Read the StdDev file to get bounds
                    std_dev_df = pd.read_csv(std_dev_file_path, index_col=0)
                    lower_bound, upper_bound = std_dev_df.loc[f'{confidence}%']

                    # Read the data file and drop unnecessary columns
                    data_df = pd.read_csv(data_file_path)[['timestamp', 'close_pct_change']]
                    
                    # Cast close_pct_change values
                    data_df['close_pct_change'] = data_df['close_pct_change'].apply(
                        lambda x: cast_to_step(x, granularity, lower_bound, upper_bound)
                    )

                    # Determine save path
                    bounded_folder = os.path.join(base_folder, 'Bounded', stock_folder)
                    ensure_directory_exists(bounded_folder)
                    save_path = os.path.join(bounded_folder, file)

                    # Save the modified DataFrame
                    data_df.to_csv(save_path, index=False)

# User inputs
base_folder = 'StockData/2023-01-01-2023-03-01'
confidence = 95
granularity = 20

# Process the files
process_files(base_folder, confidence, granularity)