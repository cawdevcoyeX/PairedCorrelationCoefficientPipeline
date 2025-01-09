import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def load_and_preprocess_data(directory, stock1, stock2):
    merged_df = None
    
    for subdir in os.listdir(directory):
        if (subdir.startswith(f"{stock1}") or subdir.startswith(f"{stock2}")) and  (subdir.endswith(f"{stock2}") or subdir.endswith(f"{stock1}")):
            subdir_path = os.path.join(directory, subdir)
            
            for file in os.listdir(subdir_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(subdir_path, file)
                    df = pd.read_csv(file_path, )#usecols=['timestamp', 'slope_EMA_2', 'intercept_EMA_2', 'pearson_correlation_EMA_2', f'{stock1}_volume', f'{stock2}_volume'])
                    
                    df = df.rename(columns={
                        'slope_EMA_2': f'{subdir}_slope',
                        'intercept_EMA_2': f'{subdir}_intercept',
                        'pearson_correlation_EMA_2': f'{subdir}_pearson_correlation',
                        f'{stock1}_volume': f'{stock1}_volume',
                        f'{stock2}_volume': f'{stock2}_volume'
                    })
                    
                    if merged_df is None:
                        merged_df = df
                    else:
                        merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')
        
    # Convert 'timestamp' column to datetime objects without timezone information
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'].str.slice(start=0, stop=19), errors='coerce')


    # Compute 'timeOfDay' in decimal hours
    merged_df['timeOfDay'] = merged_df['timestamp'].dt.hour + merged_df['timestamp'].dt.minute / 60

    # Convert 'timestamp' to Unix timestamp (seconds since the epoch) for scaling
    merged_df['timestamp_unix'] = merged_df['timestamp'].apply(lambda x: x.timestamp()).astype('float64')

    # Proceed with scaling 'timestamp_unix' and 'timeOfDay'
    timestamp_scaler = MinMaxScaler()
    merged_df['timestamp_scaled'] = timestamp_scaler.fit_transform(merged_df['timestamp_unix'].values.reshape(-1, 1))

    time_scaler = MinMaxScaler()
    merged_df['timeOfDay_scaled'] = time_scaler.fit_transform(merged_df['timeOfDay'].values.reshape(-1, 1))

    # Drop NaN values if necessary
    merged_df.dropna(inplace=True)
    print(merged_df.columns)
    return merged_df



def create_training_data(merged_df, stock1, stock2, window_len):
    feature_cols = [col for col in merged_df.columns if col not in ["timestamp", "timeOfDay","timestamp_unix"]]
    target_col = f'{stock1}_{stock2}_pearson_correlation'

    features, targets = [], []
    for i in range(len(merged_df) - window_len + 1):
        windowed_features = merged_df.iloc[i:i+window_len-1][feature_cols].values

        windowed_timestamps = merged_df.iloc[i:i+window_len-1]['timestamp']

        # Check if all timestamps in the window are at least <window_len> minutes after 9:30 am
        min_time_after_open = pd.Timestamp(windowed_timestamps.iloc[0]).replace(hour=9, minute=30) + pd.Timedelta(minutes=window_len)
        if all(pd.Timestamp(ts) >= min_time_after_open for ts in windowed_timestamps):
            features.append(windowed_features)
            targets.append(merged_df.iloc[i+window_len-1][target_col])

    features = np.array(features)
    targets = np.array(targets)

    return features, targets

def save_datasets(features, targets, save_dir, stock1, stock2):
    # Generate a timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create the directory path using stock names
    dataset_dir = os.path.join(save_dir, f"{stock1}_{stock2}")
    
    # Make the directory if it doesn't exist
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Append the timestamp to the file names
    features_file = f"features_{timestamp}.npy"
    targets_file = f"targets_{timestamp}.npy"
    
    # Save the files with the timestamped names
    np.save(os.path.join(dataset_dir, features_file), features)
    np.save(os.path.join(dataset_dir, targets_file), targets)
    
    print(f"Datasets saved successfully in {dataset_dir} with timestamp {timestamp}")


def load_datasets(save_dir, stock1, stock2):
    dataset_dir = os.path.join(save_dir, f"{stock1}_{stock2}")
    features = np.load(os.path.join(dataset_dir, "features.npy"), allow_pickle=True)
    targets = np.load(os.path.join(dataset_dir, "targets.npy"), allow_pickle=True)
    return features, targets

# User inputs
starting_directory = "StockData/2022-01-01-2024-02-08/Pairs/"
stock1 = "AAPL"
stock2 = "MSFT"
window_len = 25
save_dir = "StockData/2022-01-01-2024-02-08/TrainingData"

# Load and preprocess data
merged_df = load_and_preprocess_data(starting_directory, stock1, stock2)

# Check if data was loaded successfully
if merged_df is not None and not merged_df.empty:
    # Create training data
    features, targets = create_training_data(merged_df, stock1, stock2, window_len)

    # Save the datasets
    save_datasets(features, targets, save_dir, stock1, stock2)

    # Optional: Load the datasets for verification
    loaded_features, loaded_targets = load_datasets(save_dir, stock1, stock2)
    print("Loaded features shape:", loaded_features.shape)
    print("Loaded targets shape:", loaded_targets.shape)
else:
    print("No data found for the given stock pair or directory.")

